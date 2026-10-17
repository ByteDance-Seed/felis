// Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: CC-BY-NC-4.0

#ifndef OPENMM_RELOCATIONMC_KERNELS_H_
#define OPENMM_RELOCATIONMC_KERNELS_H_

#include "RelocationMC.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/internal/ContextImpl.h"
#include <string>
#include <vector>
#include <sstream>

namespace OpenMM {
namespace RelocationMCPlugin {

/** Initial-box estimate for MTM whole-box candidate coverage. */
struct MtmInitialBoxPoolEstimate {
    int candidateCount;
    double sphereRadius;
    double boxVolume;
    double innerProbability;
    double expectedInnerCandidates;
    double emptyInnerProbability;
    bool valid;
};

/** Compute the probability that a whole-box candidate pool has no inner sites. */
MtmInitialBoxPoolEstimate computeMtmInitialBoxPoolEstimate(
        int candidateCount, double sphereRadius, double boxVolume);

/** Return whether the initial-box empty-inner-pool estimate exceeds one percent. */
bool shouldWarnMtmInitialBoxPool(const MtmInitialBoxPoolEstimate& estimate);

/** Format the one-time MTM initialization warning, or return an empty string. */
std::string formatMtmInitialBoxPoolWarning(const MtmInitialBoxPoolEstimate& estimate);

/**
 * This kernel is invoked by RelocationMC to propose and apply Metropolis
 * relocation moves for whole labile molecules between a ligand-centered sphere
 * and bulk solvent.
 *
 * The move changes accepted coordinates and rotates the moved molecule's
 * COM-relative velocities while preserving its mass-weighted COM velocity; it
 * neither contributes forces nor potential energy of its own.  A platform
 * implementation classifies labile reference atoms around the unique ligand
 * center, selects one source molecule with its removal-energy weight, generates
 * target-region conformations, evaluates configured interaction energies, and
 * applies the cavity/direction/partition-biased acceptance rule.
 */
class CalcRelocationMCKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "RelocationMC";
    }
    CalcRelocationMCKernel(std::string name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     *
     * @param system  the System this kernel will be applied to
     * @param mc      the RelocationMC this kernel will be used for
     * @param context the context in which the kernel will be used
     */
    virtual void initialize(const OpenMM::System& system, const RelocationMC& mc, OpenMM::ContextImpl& context) = 0;
    /**
     * Attempt one cavity-biased delete/reinsert proposal on a Context.
     *
     * A source molecule is energy-weighted within the selected region.  Candidate
     * anchors and prepared orientations are generated in the target region and
     * evaluated with the configured legacy or environment-only interaction model;
     * the ordered first-success rule selects at most one accepted conformation.
     *
     * @param context      the context in which to execute this kernel
     * @param logStream    the stream to which to write log messages
     * @return one ExchangeResult
     */
    virtual ExchangeResult attemptExchange(OpenMM::ContextImpl& context, std::stringstream& logStream) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context  the context to copy parameters to
     * @param force    the RelocationMC to copy the parameters from
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const RelocationMC& force) = 0;
};

} // namespace RelocationMCPlugin
} // namespace OpenMM

#endif /*OPENMM_RELOCATIONMC_KERNELS_H_*/
