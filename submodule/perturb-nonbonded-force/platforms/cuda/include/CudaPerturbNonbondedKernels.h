#ifndef OPENMM_CUDACALCPERTURBNONBONDEDFORCEKERNEL_H_
#define OPENMM_CUDACALCPERTURBNONBONDEDFORCEKERNEL_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2008-2025 Stanford University and the Authors.      *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * This program is free software: you can redistribute it and/or modify       *
 * it under the terms of the GNU Lesser General Public License as published   *
 * by the Free Software Foundation, either version 3 of the License, or       *
 * (at your option) any later version.                                        *
 *                                                                            *
 * This program is distributed in the hope that it will be useful,            *
 * but WITHOUT ANY WARRANTY; without even the implied warranty of             *
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
 * GNU Lesser General Public License for more details.                        *
 *                                                                            *
 * You should have received a copy of the GNU Lesser General Public License   *
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
 * -------------------------------------------------------------------------- */

#include "openmm/kernels.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaEvent.h"
#include "openmm/cuda/CudaKernel.h"
#include "openmm/cuda/CudaQueue.h"
#include "openmm/cuda/CudaSort.h"
#include "openmm/cuda/CudaFFT3D.h"
#include "openmm/cuda/CudaProgram.h"
#include "openmm/common/ComputeParameterInfo.h"
#include "PerturbNonbondedKernels.h"
#include "PerturbNonbondedForce.h"
#include "internal/PerturbNonbondedForceImpl.h"
#include <array>
#include <map>
#include <string>
#include <utility>
#include <vector>

using namespace OpenMM;

namespace PerturbNonbondedPlugin {

/**
 * This kernel is invoked by PerturbNonbondedForce to calculate the forces acting on the system.
 */
class CudaCalcPerturbNonbondedForceKernel : public CalcPerturbNonbondedForceKernel {
public:
    CudaCalcPerturbNonbondedForceKernel(std::string name, const Platform& platform, CudaContext& cu, const System& system) : CalcPerturbNonbondedForceKernel(name, platform),
            hasInitializedKernel(false), cu(cu), pmeio(NULL), stepsToSort(0) {
    }
    ~CudaCalcPerturbNonbondedForceKernel();
    /**
     * Initialize the kernel.
     *
     * @param system     the System this kernel will be applied to
     * @param force      the PerturbNonbondedForce this kernel will be used for
     */
    void initialize(const System& system, const PerturbNonbondedForce& force);
    /**
     * Initialize the kernel.  Subclasses should call this from their initialize() method.
     *
     * @param system       the System this kernel will be applied to
     * @param force        the NonbondedForce this kernel will be used for
     * @param usePmeQueue  whether to perform PME on a separate queue
     * @param deviceIsCpu  whether the device this calculation is running on is a CPU
     * @param useFixedPointChargeSpreading  whether PME charge spreading should be done in fixed point or floating point
     * @param useCpuPme    whether to perform the PME reciprocal space calculation on the CPU
     */
    void commonInitialize(const System& system, const PerturbNonbondedForce& force, bool usePmeQueue, bool deviceIsCpu, bool useFixedPointChargeSpreading, bool useCpuPme);
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
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy, bool includeDirect, bool includeReciprocal);
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
    void copyParametersToContext(ContextImpl& context, const PerturbNonbondedForce& force, int firstParticle, int lastParticle, int firstException, int lastException);
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
     * Get the parameters being used for the dispersion term in LJPME.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;

    vector<double> getPerturbDispersionCorrections();

    double calcCachedPerturbDispersionCorrection(
        std::vector<std::array<double, 4>> sigmalEpsilonParams, 
        std::vector<std::array<bool, 2>> particleTypes, 
        double rest2_scale, double lambdaA_vdw, double lambdaB_vdw);
    double calcCachedEwaldSelfEnergy(
        std::vector<std::array<double, 3>> chargesParams,
        std::vector<std::array<bool, 2>> particleTypes, 
        double lambdaA, double lambdaB, double lambdaC, double rest2_scale);

    double calcDispersionCorrection(std::vector<double> sigmaList, std::vector<double> epsilonList);

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
        useParticleQueryBlockPruning = enabled ? 1 : 0;
    }
private:
    class SortTrait : public ComputeSortImpl::SortTrait {
        int getDataSize() const {return 8;}
        int getKeySize() const {return 4;}
        const char* getDataType() const {return "int2";}
        const char* getKeyType() const {return "int";}
        const char* getMinKey() const {return "(-2147483647-1)";}
        const char* getMaxKey() const {return "2147483647";}
        const char* getMaxValue() const {return "make_int2(2147483647, 2147483647)";}
        const char* getSortKey() const {return "value.y";}
    };
    class ForceInfo;
    class PmeIO;
    class PmePreComputation;
    class PmePostComputation;
    class SyncQueuePreComputation;
    class SyncQueuePostComputation;
    CudaContext& cu;
    ForceInfo* info;
    bool hasInitializedKernel;
    CudaArray charges;
    CudaArray sigmaEpsilon;
    CudaArray exceptionParams;
    CudaArray exclusionAtoms;
    CudaArray exclusionParams;
    CudaArray baseParticleParams;
    CudaArray baseExceptionParams;
    CudaArray particleParamOffsets;
    CudaArray exceptionParamOffsets;
    CudaArray particleOffsetIndices;
    CudaArray exceptionOffsetIndices;
    CudaArray cosSinSums;
    CudaArray pmeGrid1;
    CudaArray pmeGrid2;
    CudaArray scratchPmeGrid1;
    CudaArray scratchPmeGrid2;
    CudaArray scratchPmePotentials;
    CudaArray pmeBsplineModuliX;
    CudaArray pmeBsplineModuliY;
    CudaArray pmeBsplineModuliZ;
    CudaArray pmeDispersionBsplineModuliX;
    CudaArray pmeDispersionBsplineModuliY;
    CudaArray pmeDispersionBsplineModuliZ;
    CudaArray pmeAtomGridIndex;
    CudaArray pmeEnergyBuffer;
    CudaArray chargeBuffer;
    ComputeSort sort;
    ComputeQueue pmeQueue;
    ComputeEvent pmeSyncEvent, paramsSyncEvent;
    FFT3D fft, dispersionFft;
    Kernel cpuPme;
    PmeIO* pmeio;
    SyncQueuePostComputation* syncQueue;
    ComputeKernel computeParamsKernel, computeExclusionParamsKernel, computePlasmaCorrectionKernel;
    ComputeKernel ewaldSumsKernel, ewaldForcesKernel;
    ComputeKernel pmeGridIndexKernel, pmeDispersionGridIndexKernel;
    ComputeKernel pmeSpreadChargeKernel, pmeDispersionSpreadChargeKernel;
    ComputeKernel pmeFinishSpreadChargeKernel, pmeDispersionFinishSpreadChargeKernel;
    ComputeKernel pmeConvolutionKernel, pmeDispersionConvolutionKernel;
    ComputeKernel pmeEvalEnergyKernel, pmeDispersionEvalEnergyKernel;
    ComputeKernel pmeInterpolateForceKernel, pmeDispersionInterpolateForceKernel;
    CUfunction scratchPmeSpreadKernel;
    CUfunction reduceCandidateSelfEnergyKernel;
    ComputeKernel scratchPmeConvolutionKernel;
    std::map<std::string, std::string> pmeDefines;
    std::vector<std::pair<int, int> > exceptionAtoms;
    std::vector<std::string> paramNames;
    std::map<std::string, int> paramIndices;
    std::map<std::string, double> paramValues;
    std::map<int, int> exceptionIndex;
    double ewaldSelfEnergy, dispersionCoefficient, alpha, dispersionAlpha, totalCharge;
    int gridSizeX, gridSizeY, gridSizeZ;
    int dispersionGridSizeX, dispersionGridSizeY, dispersionGridSizeZ;
    int stepsToSort;
    bool usePmeQueue, deviceIsCpu, useFixedPointChargeSpreading, useCpuPme;
    bool hasCoulomb, hasLJ, doLJPME, usePosqCharges, recomputeParams, hasOffsets;
    int useParticleQueryBlockPruning = 1;
    PerturbNonbondedMethod nonbondedMethod;
    static const int PmeOrder = 5;

    // Parameters that fetch from the context, globally controlled.
    double lambdaA_charge, lambdaB_charge, lambdaC_charge;
    bool perturbParamsUpdateFlag = true;
    double lambdaA_vdw, lambdaB_vdw;
    double rest2_scale;

    double lambdaA_charge_old, lambdaB_charge_old, lambdaC_charge_old;
    double lambdaA_vdw_old, lambdaB_vdw_old;
    double rest2_scale_old;

    // If the PerturbNonbondedForce enabled constant Energy cache, 
    //then we need to cache the ewaldSelfEnergy and Ecorr corresponding to the current lambda values.
    bool enableCacheEnergyConstants= false;
    int replica_id = -1, replica_id_old = -1;
    int numReplicas = 0;
    map<int, double> ewaldSelfEnergyCache;
    map<int, double> EcorrCache;
    map<string, vector<double>> lambdaSchedules;

    std::vector<std::array<double, 3>> perturbChargesParams_cpu; // chargeA, chargeB, chargeC
    std::vector<std::array<double, 4>> perturbSigmaEpsilonParams_cpu; // sigmaA, epsilonA, sigmaB, epsilonB
    std::vector<std::array<bool, 2>> perturbAtomTypesParams_cpu; // isDummy, isREST2
    int numParticles;
    bool useSwitchingFunction;
    double cutoff = 0.0;
    double switchDist = 0.0;

    double softcore_alpha;
    double EcorrA = 0.0, EcorrB = 0.0;
    bool usePerturbDispersionCorrection;
    CudaArray perturbChargeParams;
    CudaArray perturbSigmaEpsilonParams;
    CudaArray ljAtomTypes;
    CudaArray sigmaEpsilonA;
    CudaArray sigmaEpsilonB;
    CudaArray ljLambdaAB;

    ComputeKernel updatePerturbParamsKernel;

    CUfunction realParticleEnergyKernel;
    CUfunction candidateParticleEnergyKernel;
    CUfunction candidateParticleEnergyContextPrecisionKernel;
    CUfunction candidateSubsetEnergyKernel;
    CUfunction candidateSubsetEnergyContextPrecisionKernel;
    CUfunction realPmeDirectEnergyKernel;
    CUfunction pmePotentialKernel;
    CUfunction pmePotentialContextPrecisionKernel;
    CUfunction scratchPmeSpreadContextPrecisionKernel;
    CUfunction pmeAtomicEnergyKernel;
    CUfunction sumMixedValuesKernel;
};

} // namespace PerturbNonbondedPlugin

#endif /*OPENMM_CUDACALCPERTURBNONBONDEDFORCEKERNEL_H_*/
