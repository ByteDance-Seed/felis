// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: CC-BY-NC-4.0

#ifndef OPENMM_RELOCATIONMC_H_
#define OPENMM_RELOCATIONMC_H_

#include "internal/windowsExportRelocationMC.h"
#include <openmm/Context.h>
#include <openmm/Force.h>
#include <openmm/Vec3.h>
#include <cstdint>
#include <string>
#include <vector>

namespace OpenMM {
class ForceImpl;
namespace RelocationMCPlugin {

class RelocationMCImpl;

/** Nonbonded parameters for one site of a labile molecule.
 *
 * Charge is in elementary-charge units, sigma is in nm, and epsilon is in
 * kJ/mol.  All registered labile molecules use this same site ordering.
 */
struct OPENMM_EXPORT_RELOCATIONMC LabileMolSiteParameters {
    LabileMolSiteParameters(double charge = 0.0, double sigma = 0.0, double epsilon = 0.0) :
            charge(charge), sigma(sigma), epsilon(epsilon) {
    }
    double charge;
    double sigma;
    double epsilon;
};

/** A complete molecule that may be relocated as one rigid coordinate set. */
struct OPENMM_EXPORT_RELOCATIONMC LabileMol {
    std::vector<int> atomIndices;
    int referenceAtom;
};

/** Result of one cavity-biased relocation attempt. */
struct OPENMM_EXPORT_RELOCATIONMC ExchangeResult {
    bool attempted = false;
    bool accepted = false;
    bool exchangeToInner = false;
    int movedLabileMol = -1;
    double deletedMoleculeEnergy = 0.0;
    double candidateMoleculeEnergy = 0.0;
    double deletionProbability = 0.0;
    // Legacy stores the selected per-candidate probability (or the maximum
    // examined probability on rejection).  MTM maps this compatibility field to
    // the final acceptanceProbability.
    double insertionProbability = 0.0;
    // Compact protocol diagnostics.  protocolVersion is 0 until a platform
    // implementation identifies the protocol used for this result.
    int protocolVersion = 0;
    double candidateSelectionProbability = 0.0;
    double acceptanceProbability = 0.0;
    double deletedSourceScore = 0.0;
    double candidateSourceScore = 0.0;
    double sourceLogSum = 0.0;
    double proposedSourceLogSum = 0.0;
    double forwardLogWeightSum = 0.0;
    double reverseLogWeightSum = 0.0;
    double logAcceptanceRatio = 0.0;
    double energyDelta = 0.0;
    // Correction beyond u_C-u_D used by the acceptance ratio.  In the supported
    // PME implementation this is the isolatedReciprocalDelta measured below.
    double energyResidual = 0.0;
    // 0.5*(r(C)-r(D)), where r is the isolated-molecule reciprocal mesh
    // contribution.  This is not the analytic Ewald self term: controlled
    // Context-energy oracles show it is the finite-grid correction required by
    // the current numerical Hamiltonian.
    double isolatedReciprocalDelta = 0.0;
    // PME environment-interaction diagnostic components for the selected move.
    // These are filled by the CUDA implementation after candidate selection.
    double deletedDirectEnergy = 0.0;
    double deletedFullMeshCrossEnergy = 0.0;
    double deletedMeshSelfEnergy = 0.0;
    double deletedMeshRemovalEnergy = 0.0;
    double candidateEnvironmentDirectEnergy = 0.0;
    double candidateFullMeshCrossEnergy = 0.0;
    double candidateDeletedMeshCrossEnergy = 0.0;
    // In environment-interaction mode this stores candidate reciprocal
    // interaction energy (full-grid cross minus deleted-grid cross).
    double candidateMeshSelfEnergy = 0.0;
    std::vector<Vec3> newPositions;
    std::vector<int> movedAtomIndices;
};

/**
 * Relocate complete labile molecules between a ligand-centred sphere and bulk.
 *
 * A move selects a real molecule in one region, samples candidate rigid-body
 * conformations in the other region, and applies the accepted conformation to
 * the selected molecule.  On acceptance, the molecule's mass-weighted COM
 * velocity is preserved while its COM-relative velocities are rotated by the
 * proper best-fit rotation from source to actual accepted candidate coordinates.
 * Rejected moves and non-moved atoms retain their velocities.  Only orthorhombic
 * periodic boxes are supported.  The Force itself contributes no force or energy.
 */
class OPENMM_EXPORT_RELOCATIONMC RelocationMC : public Force {
public:
    /**
     * Select the relocation proposal and acceptance protocol.
     *
     * Legacy preserves target-region sampling, prepared absolute orientations,
     * and ordered first-success acceptance.  MTM selects the paired whole-box
     * multiple-try Metropolis protocol.  The protocol is a Context construction
     * setting, so changing it requires Context reinitialization.
     */
    enum ProtocolVersion {
        Legacy = 1,
        MTM = 2
    };

    explicit RelocationMC(double temperature = 300.0, double sphereRadius = 0.5,
            int exchangeFrequency = 100, int numCandidateSitesPerExchange = 300);

    /**
     * Set the proposal protocol.  New Forces default to MTM.  Legacy is retained
     * only for explicit trajectory compatibility.  MTM requires orientationsEnabled=true,
     * useCavityBias=false, and useLegacyAsymmetricEnergies=false.
     */
    void setProtocolVersion(ProtocolVersion protocol);
    ProtocolVersion getProtocolVersion() const;

    void setTemperature(double temperature);
    double getTemperature() const;

    void setSphereRadius(double radius);
    double getSphereRadius() const;

    /** Number of integration steps between automatic relocation attempts. */
    void setExchangeFrequency(int frequency);
    int getExchangeFrequency() const;

    /**
     * Set sampled anchor sites per attempt.  Legacy expands each site to all 60
     * prepared orientations when orientations are enabled (K=sites*60).  MTM
     * interprets this value directly as K independent whole-box anchor and Haar
     * orientation draws.
     */
    void setNumCandidateSitesPerExchange(int numCandidateSites);
    int getNumCandidateSitesPerExchange() const;

    void setRandomSeed(uint64_t seed);
    uint64_t getRandomSeed() const;

    void setLigandAtoms(const std::vector<int>& ligandAtoms);
    const std::vector<int>& getLigandAtoms() const;

    void setLabileMolSiteParameters(const std::vector<LabileMolSiteParameters>& parameters);
    const std::vector<LabileMolSiteParameters>& getLabileMolSiteParameters() const;

    void addLabileMol(const std::vector<int>& atomIndices, int referenceAtom);
    int getNumLabileMols() const;
    void getLabileMolParameters(int index, std::vector<int>& atomIndices, int& referenceAtom) const;
    const std::vector<LabileMol>& getLabileMols() const;
    int getNumAtomsPerLabileMol() const;

    /** Reference coordinates relative to the candidate anchor point. */
    void setLabileMolReferencePositions(const std::vector<Vec3>& positions);
    const std::vector<Vec3>& getLabileMolReferencePositions() const;

    /**
     * Enable the prepared orientation set for Legacy.  MTM always uses independent
     * Haar orientations and therefore requires this setting to remain true.
     * Changing it requires Context reinitialization.
     */
    void setOrientationsEnabled(bool enabled);
    bool getOrientationsEnabled() const;

    /** Enable the Legacy explicit region-volume correction.  MTM requires false. */
    void setUseCavityBias(bool enabled);
    bool getUseCavityBias() const;

    /**
     * Select the historical asymmetric RF-delete/PME-insert energy estimator.
     * This compatibility mode preserves old trajectories but includes deleted
     * molecules in candidate PME grids.  The default exact mode removes that
     * contamination and uses PME direct/reciprocal interactions with the common
     * environment (the selected deleted molecule removed).
     */
    void setUseLegacyAsymmetricEnergies(bool enabled);
    bool getUseLegacyAsymmetricEnergies() const;

    void setVerbose(bool verbose);
    bool getVerbose() const;
    void setLogFileName(const std::string& fileName);
    const std::string& getLogFileName() const;

    /** Return whether an automatic attempt is due at the Context's positive current step. */
    bool isExchangeDue(const Context& context) const;

    /** Perform one relocation immediately, independent of exchangeFrequency. */
    ExchangeResult attemptExchange(Context& context);

    /** Copy mutable scalar settings to an existing Context. */
    void updateParametersInContext(Context& context);

    bool usesPeriodicBoundaryConditions() const override {
        return true;
    }

protected:
    ForceImpl* createImpl() const override;

private:
    double temperature;
    double sphereRadius;
    int exchangeFrequency;
    int numCandidateSitesPerExchange;
    ProtocolVersion protocolVersion;
    uint64_t randomSeed;
    std::vector<int> ligandAtoms;
    std::vector<LabileMolSiteParameters> labileMolSiteParameters;
    std::vector<LabileMol> labileMols;
    std::vector<Vec3> labileMolReferencePositions;
    bool orientationsEnabled;
    bool useCavityBias;
    bool useLegacyAsymmetricEnergies;
    bool verbose;
    std::string logFileName;
};

} // namespace RelocationMCPlugin
} // namespace OpenMM

#endif
