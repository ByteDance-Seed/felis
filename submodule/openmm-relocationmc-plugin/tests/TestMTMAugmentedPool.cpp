// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: CC-BY-NC-4.0

#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using std::vector;

namespace {

struct Model {
    vector<double> target;
    vector<double> h0;
    vector<int> region;
    int poolSize;
};

void require(bool condition, const std::string& message) {
    if (!condition)
        throw std::runtime_error(message);
}

void requireNear(double expected, double actual, double tolerance, const std::string& message) {
    if (!std::isfinite(expected) || !std::isfinite(actual) || std::abs(expected-actual) > tolerance)
        throw std::runtime_error(message+": expected "+std::to_string(expected)+", found "+std::to_string(actual));
}

void validate(const Model& model) {
    const int n = (int) model.target.size();
    require(n > 1, "The oracle needs at least two states.");
    require((int) model.h0.size() == n && (int) model.region.size() == n, "Model arrays have inconsistent sizes.");
    require(model.poolSize > 0, "Pool size must be positive.");
    double piSum = 0.0;
    double h0Sum = 0.0;
    for (int i = 0; i < n; i++) {
        require(std::isfinite(model.target[i]) && model.target[i] > 0.0, "Target masses must be positive and finite.");
        require(std::isfinite(model.h0[i]) && model.h0[i] > 0.0, "h0 masses must be positive and finite.");
        require(model.region[i] == 0 || model.region[i] == 1, "The oracle has exactly two regions.");
        piSum += model.target[i];
        h0Sum += model.h0[i];
    }
    requireNear(1.0, piSum, 1e-14, "Target masses are not normalized");
    requireNear(1.0, h0Sum, 1e-14, "h0 masses are not normalized");
}

template <class Callback>
void enumeratePools(const Model& model, Callback callback) {
    vector<int> pool(model.poolSize, 0);
    std::function<void(int, double)> recurse = [&](int slot, double probability) {
        if (slot == model.poolSize) {
            callback(pool, probability);
            return;
        }
        for (int state = 0; state < (int) model.h0.size(); state++) {
            pool[slot] = state;
            recurse(slot+1, probability*model.h0[state]);
        }
    };
    recurse(0, 1.0);
}

vector<double> makeWeights(const Model& model, bool includeProposalDensity) {
    vector<double> weights(model.target.size());
    for (int i = 0; i < (int) weights.size(); i++)
        weights[i] = includeProposalDensity ? model.target[i]/model.h0[i] : model.target[i];
    return weights;
}

vector<vector<double> > buildTransition(const Model& model, bool includeProposalDensity = true) {
    validate(model);
    const int n = (int) model.target.size();
    const vector<double> weights = makeWeights(model, includeProposalDensity);
    vector<vector<double> > transition(n, vector<double>(n, 0.0));
    for (int x = 0; x < n; x++) {
        enumeratePools(model, [&](const vector<int>& pool, double poolProbability) {
            vector<int> eligible;
            double forwardSum = 0.0;
            for (int slot = 0; slot < model.poolSize; slot++) {
                if (model.region[pool[slot]] != model.region[x]) {
                    eligible.push_back(slot);
                    forwardSum += weights[pool[slot]];
                }
            }
            if (eligible.empty()) {
                transition[x][x] += poolProbability;
                return;
            }
            for (int selectedSlot : eligible) {
                const int y = pool[selectedSlot];
                const double selection = weights[y]/forwardSum;
                vector<int> reversePool = pool;
                reversePool[selectedSlot] = x;
                double reverseSum = 0.0;
                for (int state : reversePool)
                    if (model.region[state] == model.region[x])
                        reverseSum += weights[state];
                require(reverseSum > 0.0, "The reverse pool did not contain the old state.");
                const double acceptance = std::min(1.0, forwardSum/reverseSum);
                const double eventMass = poolProbability*selection;
                transition[x][y] += eventMass*acceptance;
                transition[x][x] += eventMass*(1.0-acceptance);
            }
        });
    }
    return transition;
}

double maximumDetailedBalanceError(const Model& model, const vector<vector<double> >& transition) {
    double error = 0.0;
    for (int x = 0; x < (int) model.target.size(); x++)
        for (int y = 0; y < (int) model.target.size(); y++)
            error = std::max(error, std::abs(model.target[x]*transition[x][y]-model.target[y]*transition[y][x]));
    return error;
}

void checkTransition(const Model& model, const vector<vector<double> >& transition) {
    for (int x = 0; x < (int) model.target.size(); x++) {
        double rowSum = 0.0;
        for (double value : transition[x]) {
            require(value >= -1e-15 && value <= 1.0+1e-15, "Transition probability is outside [0,1].");
            rowSum += value;
        }
        requireNear(1.0, rowSum, 2e-14, "Transition row is not normalized");
    }
    require(maximumDetailedBalanceError(model, transition) < 3e-14, "Paired-pool kernel violates detailed balance.");
    for (int y = 0; y < (int) model.target.size(); y++) {
        double propagated = 0.0;
        for (int x = 0; x < (int) model.target.size(); x++)
            propagated += model.target[x]*transition[x][y];
        requireNear(model.target[y], propagated, 3e-14, "Target distribution is not stationary");
    }
}

Model nonuniformModel(int poolSize) {
    return {{0.10, 0.20, 0.25, 0.45},
            {0.45, 0.10, 0.30, 0.15},
            {0, 0, 1, 1}, poolSize};
}

void testEventInvolution() {
    const Model model = nonuniformModel(3);
    const vector<double> weights = makeWeights(model, true);
    for (int x = 0; x < (int) model.target.size(); x++) {
        enumeratePools(model, [&](const vector<int>& pool, double poolProbability) {
            double forwardSum = 0.0;
            for (int state : pool)
                if (model.region[state] != model.region[x])
                    forwardSum += weights[state];
            if (forwardSum == 0.0)
                return;
            for (int slot = 0; slot < model.poolSize; slot++) {
                const int y = pool[slot];
                if (model.region[y] == model.region[x])
                    continue;
                vector<int> reversePool = pool;
                reversePool[slot] = x;
                double reversePoolProbability = 1.0;
                double reverseSum = 0.0;
                for (int state : reversePool) {
                    reversePoolProbability *= model.h0[state];
                    if (model.region[state] == model.region[x])
                        reverseSum += weights[state];
                }
                const double directDensityRatio =
                        (model.target[y]/model.target[x])*
                        (reversePoolProbability/poolProbability)*
                        ((weights[x]/reverseSum)/(weights[y]/forwardSum));
                requireNear(forwardSum/reverseSum, directDensityRatio, 3e-14,
                        "Pool, categorical, and target density ratios did not reduce to Wf/Wr");
                const double forwardFlow = model.target[x]*poolProbability*(weights[y]/forwardSum)*
                        std::min(1.0, directDensityRatio);
                const double reverseFlow = model.target[y]*reversePoolProbability*(weights[x]/reverseSum)*
                        std::min(1.0, 1.0/directDensityRatio);
                requireNear(forwardFlow, reverseFlow, 3e-14, "The augmented event and its involution have unequal flow");
            }
        });
    }
}

void testK1Reduction() {
    const Model model = nonuniformModel(1);
    const vector<double> weights = makeWeights(model, true);
    const auto transition = buildTransition(model);
    checkTransition(model, transition);
    for (int x = 0; x < (int) model.target.size(); x++)
        for (int y = 0; y < (int) model.target.size(); y++) {
            if (x == y || model.region[x] == model.region[y])
                continue;
            const double expected = model.h0[y]*std::min(1.0, weights[y]/weights[x]);
            requireNear(expected, transition[x][y], 2e-14, "K=1 did not reduce to ordinary independence MH");
        }
}

void testTwoStateAnalyticalCase() {
    const Model model = {{1.0/3.0, 2.0/3.0}, {0.5, 0.5}, {0, 1}, 2};
    const auto transition = buildTransition(model);
    checkTransition(model, transition);
    requireNear(0.25, transition[0][0], 2e-14, "Unexpected A->A probability");
    requireNear(0.75, transition[0][1], 2e-14, "Unexpected A->B probability");
    requireNear(0.375, transition[1][0], 2e-14, "Unexpected B->A probability");
    requireNear(0.625, transition[1][1], 2e-14, "Unexpected B->B probability");
    const double stationaryA = transition[1][0]/(transition[0][1]+transition[1][0]);
    requireNear(1.0/3.0, stationaryA, 2e-14, "Two-state stationary distribution is incorrect");
}

void testNonuniformProposalNeedsInverseH0() {
    const Model model = nonuniformModel(3);
    const auto exact = buildTransition(model, true);
    checkTransition(model, exact);
    const auto wrong = buildTransition(model, false);
    require(maximumDetailedBalanceError(model, wrong) > 1e-3,
            "The nonuniform-h0 control did not detect omission of the 1/h0 importance factor.");
}

double bilinear(const vector<double>& left, const vector<vector<double> >& kernel,
        const vector<double>& right) {
    double value = 0.0;
    for (int i = 0; i < (int) left.size(); i++)
        for (int j = 0; j < (int) right.size(); j++)
            value += left[i]*kernel[i][j]*right[j];
    return value;
}

vector<double> add(const vector<double>& first, const vector<double>& second) {
    vector<double> result(first.size());
    for (int i = 0; i < (int) result.size(); i++)
        result[i] = first[i]+second[i];
    return result;
}

void testK1AcceptanceFactorBookkeeping() {
    const double beta = 0.7;
    const double uDeleted = -3.2;
    const double uCandidate = 1.4;
    const double rDeleted = 0.3;
    const double rCandidate = -0.2;
    const double logDirectionRatio = std::log(0.5);
    const double logPartitionRatio = std::log(1.7);
    const double sourceDifference = (uCandidate+rCandidate)-(uDeleted+rDeleted);
    const double logForwardWeight = -beta*uCandidate;
    const double logReverseWeight = -beta*uDeleted;
    const double energyResidual = 0.0;
    const double implemented = logDirectionRatio+logPartitionRatio+beta*sourceDifference+
            logForwardWeight-logReverseWeight-beta*energyResidual;
    const double expected = logDirectionRatio+logPartitionRatio+beta*(rCandidate-rDeleted);
    requireNear(expected, implemented, 2e-15,
            "K=1 acceptance double-counted the common-environment energy difference");
}

void testReciprocalQuadraticDecomposition() {
    // A finite symmetric operator models the exact linear reciprocal solver,
    // including periodic-image and mesh/interpolation effects.  D and C carry
    // the same charge multiset at different grid coordinates/orientations.
    const vector<vector<double> > reciprocal = {
        {1.7, -0.2, 0.1, 0.3}, {-0.2, 1.2, 0.4, -0.1},
        {0.1, 0.4, 1.5, 0.2}, {0.3, -0.1, 0.2, 1.9}};
    const vector<double> background = {0.6, -0.4, 0.2, 0.1};
    const vector<double> deleted = {0.8, -0.3, 0.0, 0.0};
    const vector<double> candidate = {0.0, 0.0, -0.3, 0.8};
    const vector<double> stateX = add(background, deleted);
    const vector<double> stateY = add(background, candidate);
    const double fullDelta = 0.5*bilinear(stateY, reciprocal, stateY)-
            0.5*bilinear(stateX, reciprocal, stateX);
    const double uDeleted = bilinear(deleted, reciprocal, background);
    const double uCandidate = bilinear(candidate, reciprocal, background);
    const double deletedIsolatedCross = bilinear(deleted, reciprocal, deleted);
    const double candidateIsolatedCross = bilinear(candidate, reciprocal, candidate);
    const double isolatedReciprocalDelta = 0.5*(candidateIsolatedCross-deletedIsolatedCross);
    requireNear(fullDelta, uCandidate-uDeleted+isolatedReciprocalDelta, 2e-14,
            "Reciprocal quadratic expansion has the wrong cross or one-half coefficient");

    // Source selection uses q*psi_full, hence one complete isolated cross,
    // while the physical quadratic-energy residual uses one half of its change.
    const double sourceDeleted = bilinear(deleted, reciprocal, stateX);
    const double sourceCandidate = bilinear(candidate, reciprocal, stateY);
    requireNear(uDeleted+deletedIsolatedCross, sourceDeleted, 2e-14,
            "Deleted source score omitted or double-counted its isolated reciprocal cross");
    requireNear(uCandidate+candidateIsolatedCross, sourceCandidate, 2e-14,
            "Candidate source score omitted or double-counted its isolated reciprocal cross");

    // These terms depend only on the unchanged charge multiset, net charge, or
    // rigid internal geometry and therefore cancel separately under relocation.
    const double alpha = 0.31;
    double deletedChargeSquare = 0.0;
    double candidateChargeSquare = 0.0;
    double deletedNetCharge = 0.0;
    double candidateNetCharge = 0.0;
    for (double charge : deleted) {
        deletedChargeSquare += charge*charge;
        deletedNetCharge += charge;
    }
    for (double charge : candidate) {
        candidateChargeSquare += charge*charge;
        candidateNetCharge += charge;
    }
    const double analyticSelfDeleted = -alpha*deletedChargeSquare;
    const double analyticSelfCandidate = -alpha*candidateChargeSquare;
    const double netCorrectionDeleted = -0.17*deletedNetCharge*deletedNetCharge;
    const double netCorrectionCandidate = -0.17*candidateNetCharge*candidateNetCharge;
    const double zeroExceptionCorrectionDeleted = 0.23; // same rigid internal distances
    const double zeroExceptionCorrectionCandidate = 0.23;
    requireNear(analyticSelfDeleted, analyticSelfCandidate, 0.0,
            "Analytic Ewald self correction did not cancel for an unchanged charge multiset");
    requireNear(netCorrectionDeleted, netCorrectionCandidate, 0.0,
            "Net-charge correction did not cancel for an unchanged molecular charge");
    requireNear(zeroExceptionCorrectionDeleted, zeroExceptionCorrectionCandidate, 0.0,
            "Rigid zero-exception correction did not cancel");

    // If the isolated reciprocal contribution is invariant, the general
    // expression reduces to DeltaU=u_C-u_D.  Otherwise the measured half
    // difference is part of the implemented Hamiltonian and must be retained.
    const vector<double> invariantCandidate = deleted;
    const double invariantDelta = 0.5*(bilinear(invariantCandidate, reciprocal, invariantCandidate)-
            bilinear(deleted, reciprocal, deleted));
    requireNear(0.0, invariantDelta, 0.0,
            "Invariant isolated reciprocal contributions should give zero residual");
}

void testNonselectedPointsAreConditionallyBiased() {
    // Condition on x=state 0 and on slot 0 being categorically selected as state 2.
    // Before selection slot 1 is h0-distributed.  Afterwards its posterior includes
    // w(2)/(w(2)+1[target(slot1)]w(slot1)), so it is not h0-distributed.
    const Model model = nonuniformModel(2);
    const vector<double> weights = makeWeights(model, true);
    const int x = 0;
    const int selectedState = 2;
    vector<double> posterior(model.target.size(), 0.0);
    double normalization = 0.0;
    for (int other = 0; other < (int) model.target.size(); other++) {
        double forwardSum = weights[selectedState];
        if (model.region[other] != model.region[x])
            forwardSum += weights[other];
        const double jointMass = model.h0[selectedState]*model.h0[other]*weights[selectedState]/forwardSum;
        posterior[other] = jointMass;
        normalization += jointMass;
    }
    for (double& value : posterior)
        value /= normalization;
    double maximumDifference = 0.0;
    for (int state = 0; state < (int) posterior.size(); state++)
        maximumDifference = std::max(maximumDifference, std::abs(posterior[state]-model.h0[state]));
    require(maximumDifference > 0.10, "Selection did not produce the expected conditional bias in nonselected points.");

    // The event-involution test above nevertheless balances every complete pool.
    // Keeping the realized nonselected points and retaining both categorical
    // factors is what accounts for this conditional bias.
    requireNear(0.6142433234421365, posterior[0], 2e-14, "Unexpected conditional posterior for the retained point");
}

} // namespace

int main() {
    try {
        testEventInvolution();
        testK1Reduction();
        testTwoStateAnalyticalCase();
        testNonuniformProposalNeedsInverseH0();
        testK1AcceptanceFactorBookkeeping();
        testReciprocalQuadraticDecomposition();
        testNonselectedPointsAreConditionallyBiased();
        std::cout << "Done" << std::endl;
        return 0;
    }
    catch (const std::exception& error) {
        std::cerr << "Test failed: " << error.what() << std::endl;
        return 1;
    }
}
