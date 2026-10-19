#ifndef OPENMM_PERTURBNONBONDEDFORCEIMPL_H_
#define OPENMM_PERTURBNONBONDEDFORCEIMPL_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2024 Stanford University and the Authors.      *
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

#include "openmm/internal/ForceImpl.h"
#include "PerturbNonbondedForce.h"
#include "openmm/Kernel.h"
#include <utility>
#include <set>
#include <string>
#include <map>

namespace PerturbNonbondedPlugin {

//class System;

/**
 * This is the internal implementation of PerturbNonbondedForce.
 */

class OPENMM_EXPORT_PERTURBNONBONDED PerturbNonbondedForceImpl : public ForceImpl {
public:
    PerturbNonbondedForceImpl(const PerturbNonbondedForce& owner);
    ~PerturbNonbondedForceImpl();
    void initialize(ContextImpl& context);
    const PerturbNonbondedForce& getOwner() const {
        return owner;
    }
    void updateContextState(ContextImpl& context, bool& forcesInvalid) {
        // This force field doesn't update the state directly.
    }
    double calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups);
    std::map<std::string, double> getDefaultParameters();
    std::vector<std::string> getKernelNames();
    void updateParametersInContext(ContextImpl& context, int firstParticle, int lastParticle, int firstException, int lastException);
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    void getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * This is a utility routine that calculates the values to use for alpha and kmax when using
     * Ewald summation.
     */
    static void calcEwaldParameters(const System& system, const PerturbNonbondedForce& force, double& alpha, int& kmaxx, int& kmaxy, int& kmaxz);
    /**
     * This is a utility routine that calculates the values to use for alpha and grid size when using
     * Particle Mesh Ewald.
     */
    static void calcPMEParameters(const System& system, const PerturbNonbondedForce& force, double& alpha, int& xsize, int& ysize, int& zsize, bool lj);
    /**
     * Compute the coefficient which, when divided by the periodic box volume, gives the
     * long range dispersion correction to the energy.
     */
    static double calcDispersionCorrection(const System& system, const PerturbNonbondedForce& force);

    void getInteractionEnergyForRealParticlesCuda(ContextImpl& context, int numRealAtoms,
            CudaArrayDevicePointer positions, CudaArrayDevicePointer parameters,
            CudaArrayDevicePointer excludedAtomStarts, CudaArrayDevicePointer excludedAtomCounts,
            CudaArrayDevicePointer excludedAtoms, CudaArrayDevicePointer forces,
            CudaArrayDevicePointer energies);
    void getDirectEnergyForCandidateParticlesCuda(ContextImpl& context, int numCandidateAtoms,
            CudaArrayDevicePointer positions, CudaArrayDevicePointer parameters,
            CudaArrayDevicePointer forces, CudaArrayDevicePointer energies);
    void getDirectEnergyForCandidateParticlesCudaContextPrecision(ContextImpl& context,
            int numCandidateAtoms, CudaArrayDevicePointer positionBase,
            CudaArrayDevicePointer positionCorrection, CudaArrayDevicePointer parameters,
            CudaArrayDevicePointer forces, CudaArrayDevicePointer energies);
    void getPMEPotentialAtPositionsCuda(ContextImpl& context, int numPositions,
            CudaArrayDevicePointer positions, CudaArrayDevicePointer potentials);
    void getPMEPotentialAtPositionsCudaContextPrecision(ContextImpl& context, int numPositions,
            CudaArrayDevicePointer positionBase, CudaArrayDevicePointer positionCorrection,
            CudaArrayDevicePointer potentials);
    void getSymmetricPMEReciprocalAtomicEnergiesCuda(ContextImpl& context,
            CudaArrayDevicePointer energies, CudaArrayDevicePointer sum);
    void getPMEPotentialFromParticlesCuda(ContextImpl& context, int numSourceParticles,
            CudaArrayDevicePointer sourcePositions, CudaArrayDevicePointer sourceCharges,
            int numQueryPositions, CudaArrayDevicePointer queryPositions,
            CudaArrayDevicePointer queryPotentials);
    void getPMEPotentialFromParticlesCudaContextPrecision(ContextImpl& context,
            int numSourceParticles, CudaArrayDevicePointer sourcePositionBase,
            CudaArrayDevicePointer sourcePositionCorrection, CudaArrayDevicePointer sourceCharges,
            int numQueryPositions, CudaArrayDevicePointer queryPositionBase,
            CudaArrayDevicePointer queryPositionCorrection,
            CudaArrayDevicePointer queryPotentials);
    void getPMECandidateSelfEnergiesCuda(ContextImpl& context, int numCandidates,
            int numAtomsPerCandidate, CudaArrayDevicePointer candidatePositions,
            CudaArrayDevicePointer candidateCharges, CudaArrayDevicePointer energies);
    void getPMECandidateSelfEnergiesCudaContextPrecision(ContextImpl& context,
            int numCandidates, int numAtomsPerCandidate,
            CudaArrayDevicePointer candidatePositionBase,
            CudaArrayDevicePointer candidatePositionCorrection,
            CudaArrayDevicePointer candidateCharges, CudaArrayDevicePointer energies);
    void getDirectEnergyForCandidateSubsetCuda(ContextImpl& context, int numCandidateAtoms,
            CudaArrayDevicePointer candidatePositions, CudaArrayDevicePointer candidateParameters,
            int numSourceAtoms, CudaArrayDevicePointer sourceAtomIndices,
            CudaArrayDevicePointer energies);
    void getDirectEnergyForCandidateSubsetCudaContextPrecision(ContextImpl& context,
            int numCandidateAtoms, CudaArrayDevicePointer candidatePositionBase,
            CudaArrayDevicePointer candidatePositionCorrection,
            CudaArrayDevicePointer candidateParameters, int numSourceAtoms,
            CudaArrayDevicePointer sourceAtomIndices, CudaArrayDevicePointer energies);
    void getDirectEnergyForRealParticlesPmeCuda(ContextImpl& context, int numRealAtoms,
            CudaArrayDevicePointer realAtomIndices, int numAtomsPerMolecule,
            CudaArrayDevicePointer energies);
    void setParticleQueryBlockPruning(bool enabled);
private:
    class ErrorFunction;
    class EwaldErrorFunction;
    static int findZero(const ErrorFunction& f, int initialGuess);
    static double evalIntegral(double r, double rs, double rc, double sigma);
    const PerturbNonbondedForce& owner;
    Kernel kernel;
    int recipForceGroup;
    bool includeDirectSpace;
};

} // namespace PerturbNonbondedPlugin

#endif /*OPENMM_PERTURBNONBONDEDFORCEIMPL_H_*/
