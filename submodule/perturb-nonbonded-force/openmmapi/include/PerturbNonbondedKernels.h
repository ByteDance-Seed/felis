#ifndef OPENMM_PERTURBNONBONDEDFORCES_PERTURBNONBONDEDKERNELS_H_
#define OPENMM_PERTURBNONBONDEDFORCES_PERTURBNONBONDEDKERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded under the NIH Roadmap for        *
 * Medical Research, grant U54 GM072970. See https://simtk.org.               *
 *                                                                            *
 * Portions copyright (c) 2014 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "PerturbNonbondedForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>

namespace PerturbNonbondedPlugin {

/**
 * This kernel is invoked by PerturbNonbondedForce to calculate the forces acting on the system and the energy of the system.
 */
class CalcPerturbNonbondedForceKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcPerturbNonbondedForce";
    }
    enum PerturbNonbondedMethod {
        NoCutoff = 0,
        CutoffNonPeriodic = 1,
        CutoffPeriodic = 2,
        Ewald = 3,
        PME = 4,
        LJPME = 5
    };
    CalcPerturbNonbondedForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {
    }
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the NonbondedForce this kernel will be used for
     */
    virtual void initialize(const System& system, const PerturbNonbondedForce& force) = 0;
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @param includeDirect  true if direct space interactions should be included
     * @param includeReciprocal  true if reciprocal space interactions should be included
     * @return the potential energy due to the force
     */
    virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal) = 0;
    /**
     * Copy changed parameters over to a context.
     *
     * @param context        the context to copy parameters to
     * @param force          the NonbondedForce to copy the parameters from
     * @param firstParticle  the index of the first particle whose parameters might have changed
     * @param lastParticle   the index of the last particle whose parameters might have changed
     * @param firstException the index of the first exception whose parameters might have changed
     * @param lastException  the index of the last exception whose parameters might have changed
     */
    virtual void copyParametersToContext(ContextImpl& context, const PerturbNonbondedForce& force, int firstParticle, int lastParticle, int firstException, int lastException) = 0;
    /**
     * Get the parameters being used for PME.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    virtual void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const = 0;
    /**
     * Get the parameters being used for the dispersion terms in LJPME.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    virtual void getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const = 0;

    virtual void getInteractionEnergyForRealParticlesCuda(ContextImpl& context, int numRealAtoms,
            CudaArrayDevicePointer positions, CudaArrayDevicePointer parameters,
            CudaArrayDevicePointer excludedAtomStarts, CudaArrayDevicePointer excludedAtomCounts,
            CudaArrayDevicePointer excludedAtoms, CudaArrayDevicePointer forces,
            CudaArrayDevicePointer energies) = 0;

    virtual void getDirectEnergyForCandidateParticlesCuda(ContextImpl& context, int numCandidateAtoms,
            CudaArrayDevicePointer positions, CudaArrayDevicePointer parameters,
            CudaArrayDevicePointer forces, CudaArrayDevicePointer energies) = 0;

    virtual void getPMEPotentialAtPositionsCuda(ContextImpl& context, int numPositions,
            CudaArrayDevicePointer positions, CudaArrayDevicePointer potentials) = 0;

    virtual void getSymmetricPMEReciprocalAtomicEnergiesCuda(ContextImpl& context,
            CudaArrayDevicePointer energies, CudaArrayDevicePointer sum) = 0;

    virtual void getPMEPotentialFromParticlesCuda(ContextImpl& context, int numSourceParticles,
            CudaArrayDevicePointer sourcePositions, CudaArrayDevicePointer sourceCharges,
            int numQueryPositions, CudaArrayDevicePointer queryPositions,
            CudaArrayDevicePointer queryPotentials) = 0;

    virtual void getPMECandidateSelfEnergiesCuda(ContextImpl& context, int numCandidates,
            int numAtomsPerCandidate, CudaArrayDevicePointer candidatePositions,
            CudaArrayDevicePointer candidateCharges, CudaArrayDevicePointer energies) = 0;
    virtual void getDirectEnergyForCandidateSubsetCuda(ContextImpl& context, int numCandidateAtoms,
            CudaArrayDevicePointer candidatePositions, CudaArrayDevicePointer candidateParameters,
            int numSourceAtoms, CudaArrayDevicePointer sourceAtomIndices,
            CudaArrayDevicePointer energies) = 0;
    virtual void getDirectEnergyForRealParticlesPmeCuda(ContextImpl& context, int numRealAtoms,
            CudaArrayDevicePointer realAtomIndices, int numAtomsPerMolecule,
            CudaArrayDevicePointer energies) = 0;

    virtual void setParticleQueryBlockPruning(bool enabled) = 0;

    // Keep precision-aware extensions after all legacy virtuals so the existing
    // float4 query ABI remains a stable prefix of this interface.
    virtual void getDirectEnergyForCandidateParticlesCudaContextPrecision(ContextImpl& context,
            int numCandidateAtoms, CudaArrayDevicePointer positionBase,
            CudaArrayDevicePointer positionCorrection, CudaArrayDevicePointer parameters,
            CudaArrayDevicePointer forces, CudaArrayDevicePointer energies) = 0;
    virtual void getPMEPotentialAtPositionsCudaContextPrecision(ContextImpl& context, int numPositions,
            CudaArrayDevicePointer positionBase, CudaArrayDevicePointer positionCorrection,
            CudaArrayDevicePointer potentials) = 0;
    virtual void getPMEPotentialFromParticlesCudaContextPrecision(ContextImpl& context,
            int numSourceParticles, CudaArrayDevicePointer sourcePositionBase,
            CudaArrayDevicePointer sourcePositionCorrection, CudaArrayDevicePointer sourceCharges,
            int numQueryPositions, CudaArrayDevicePointer queryPositionBase,
            CudaArrayDevicePointer queryPositionCorrection,
            CudaArrayDevicePointer queryPotentials) = 0;
    virtual void getPMECandidateSelfEnergiesCudaContextPrecision(ContextImpl& context,
            int numCandidates, int numAtomsPerCandidate,
            CudaArrayDevicePointer candidatePositionBase,
            CudaArrayDevicePointer candidatePositionCorrection,
            CudaArrayDevicePointer candidateCharges, CudaArrayDevicePointer energies) = 0;
    virtual void getDirectEnergyForCandidateSubsetCudaContextPrecision(ContextImpl& context,
            int numCandidateAtoms, CudaArrayDevicePointer candidatePositionBase,
            CudaArrayDevicePointer candidatePositionCorrection,
            CudaArrayDevicePointer candidateParameters, int numSourceAtoms,
            CudaArrayDevicePointer sourceAtomIndices, CudaArrayDevicePointer energies) = 0;
};

} // namespace PerturbNonbondedPlugin

#endif /*OPENMM_PERTURBNONBONDEDFORCES_PERTURBNONBONDEDKERNELS_H_ */
