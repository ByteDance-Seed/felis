#ifndef REFERENCE_PERTURBNONBONDED_KERNELS_H_
#define REFERENCE_PERTURBNONBONDED_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit originating from   *
 * Simbios, the NIH National Center for Physics-Based Simulation of           *
 * Biological Structures at Stanford, funded by the NIH Roadmap for        *
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

#include "PerturbNonbondedKernels.h"
#include "openmm/Platform.h"
#include "openmm/reference/ReferenceNeighborList.h"
#include <vector>
#include <array>
#include <map>
#include <set>

using namespace std;
using namespace OpenMM;

namespace PerturbNonbondedPlugin {

/**
 * This kernel is invoked by PerturbNonbondedForce to calculate the forces acting on the system and the energy of the system.
 */
class ReferenceCalcPerturbNonbondedForceKernel : public CalcPerturbNonbondedForceKernel {
public:
    ReferenceCalcPerturbNonbondedForceKernel(std::string name, const OpenMM::Platform& platform) : CalcPerturbNonbondedForceKernel(name, platform), neighborList(NULL) {
    }
    ~ReferenceCalcPerturbNonbondedForceKernel();
    /**
     * Initialize the kernel.
     * 
     * @param system     the System this kernel will be applied to
     * @param force      the PerturbNonbondedForce this kernel will be used for
     */
    void initialize(const OpenMM::System& system, const PerturbNonbondedForce& force);
    /**
     * Execute the kernel to calculate the forces and/or energy.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  true if forces should be calculated
     * @param includeEnergy  true if the energy should be calculated
     * @param includeDirect      true if direct space interactions should be included
     * @param includeReciprocal  true if reciprocal space interactions should be included
     * @return the potential energy due to the force
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal);
    /**
     * Copy changed parameters over to a context.
     *
     * @param context        the context to copy parameters to
     * @param force          the PerturbNonbondedForce to copy the parameters from
     * @param firstParticle  the index of the first particle whose parameters might have changed
     * @param lastParticle   the index of the last particle whose parameters might have changed
     * @param firstException the index of the first exception whose parameters might have changed
     * @param lastException  the index of the last exception whose parameters might have changed
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const PerturbNonbondedForce& force, int firstParticle, int lastParticle, int firstException, int lastException);
    /**
     * Get the parameters being used for PME.
     * 
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Get the dispersion parameters being used for the dispersion term in LJPME.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
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
    void setParticleQueryBlockPruning(bool enabled) {
    }

    void getPerturbDispersionCorrections(ContextImpl& context);
    double calcDispersionCorrection(const vector<double>& sigmaList, const vector<double>& epsilonList);

private:
    void computeParameters(OpenMM::ContextImpl& context);
    void computePerturbParameters(OpenMM::ContextImpl& context);

    int numParticles, num14;
    std::vector<std::vector<int> > bonded14IndexArray;
    std::vector<std::vector<double> > particleParamArray, bonded14ParamArray;
    std::vector<std::array<double, 3> > baseParticleParams, baseExceptionParams;
    std::map<std::pair<std::string, int>, std::array<double, 3> > particleParamOffsets, exceptionParamOffsets;
    std::map<int, int> nb14Index;
    double nonbondedCutoff, switchingDistance, rfDielectric, ewaldAlpha, ewaldDispersionAlpha, dispersionCoefficient;
    int kmax[3], gridSize[3], dispersionGridSize[3];
    bool useSwitchingFunction, exceptionsArePeriodic;
    std::vector<std::set<int> > exclusions;
    PerturbNonbondedForce::PerturbNonbondedMethod nonbondedMethod;
    OpenMM::NeighborList* neighborList;

    std::vector<double> particleChargesInRun;
    std::vector<std::array<double, 4> > particleSigmaEpsilonParamsInRun;

    std::vector<std::array<double, 3> > particleCoulombParams; // chargeA, chargeB, chargeC
    std::vector<std::array<double, 4> > particleSigmaEpsilonParams; // 0.5 * sigmaA, 2.0 * sqrt(epsilonA), 0.5 * sigmaB, 2.0 * sqrt(epsilonB)
    std::vector<std::array<bool, 2> > particleTypeParams; // First element is isDummy, second element is isREST2.
    std::vector<std::array<double, 4>> perturbSigmaEpsilonParams_raw; // sigmaA, epsilonA, sigmaB, epsilonB
    

    double softcore_alpha;
    double EcorrA, EcorrB;

    double lambdaA_charge, lambdaB_charge, lambdaC_charge;
    double lambdaA_vdw, lambdaB_vdw;
    double rest2_scale;
    double sqrt_rest2_scale;

    double lambdaA_charge_old = 0.0, lambdaB_charge_old = 0.0, lambdaC_charge_old = 0.0;
    double lambdaA_vdw_old = 0.0, lambdaB_vdw_old = 0.0;
    double rest2_scale_old = 0.0;

    bool updatePerturbParamsFlag;
    bool useDispersionCorrection = false;
};

} // namespace PerturbNonbondedPlugin

#endif /*REFERENCE_PERTURB_NONBONDED_KERNELS_H_*/
