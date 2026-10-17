#ifndef OPENMM_PERTURBNONBONDEDFORCE_H_
#define OPENMM_PERTURBNONBONDEDFORCE_H_

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

#include "openmm/Context.h"
#include "openmm/Force.h"
#include <map>
#include <set>
#include <utility>
#include <vector>
#include "internal/windowsExportPerturbNonbonded.h"

namespace PerturbNonbondedPlugin {

using namespace OpenMM;
using OpenMM::Context;
using OpenMM::ForceImpl;
using std::map;
using std::set;
using std::string;
using std::vector;

/** Opaque CUDA device pointer used by zero-copy energy query APIs. */
typedef unsigned long long CudaArrayDevicePointer;

/**
 * This class implements perturbed, nonbonded interactions between particles, including a Coulomb force to represent
 * electrostatics and a Lennard-Jones force to represent van der Waals interactions.  It optionally supports
 * periodic boundary conditions and cutoffs for long range interactions.  Lennard-Jones interactions are
 * calculated with the Lorentz-Berthelot combining rule: it uses the arithmetic mean of the sigmas and the
 * geometric mean of the epsilons for the two interacting particles.
 *
 * To use this class, create a PerturbNonbondedForce object, then call addParticle() once for each particle in the
 * System to define its parameters.  The number of particles for which you define nonbonded parameters must
 * be exactly equal to the number of particles in the System, or else an exception will be thrown when you
 * try to create a Context.  After a particle has been added, you can modify its force field parameters
 * by calling setParticleParameters().  This will have no effect on Contexts that already exist unless you
 * call updateParametersInContext().
 *
 * PerturbNonbondedForce also lets you specify zero-valued exceptions, particular pairs of particles whose interactions
 * are excluded.  Nonzero exception energies are not supported and should be represented by a separate bonded Force.
 *
 * Many molecular force fields omit Coulomb and Lennard-Jones interactions between particles separated by one
 * or two bonds, while using modified parameters for those separated by three bonds (known as "1-4 interactions").
 * This class provides a convenience method for this case called createExceptionsFromBonds().  You pass to it
 * a list of bonds and the scale factors to use for 1-4 interactions.  It identifies all pairs of particles which
 * are separated by 1, 2, or 3 bonds, then automatically creates exceptions for them.
 *
 * When using a cutoff, Lennard-Jones interactions are sharply truncated at the cutoff distance.  Switching functions
 * are not implemented: setUseSwitchingFunction(true) throws an exception.
 *
 * Another optional feature of this class (enabled by default) is to add a contribution to the energy which approximates
 * the effect of all Lennard-Jones interactions beyond the cutoff in a periodic system.  When running a simulation
 * at constant pressure, this can improve the quality of the result.  Call setUseDispersionCorrection() to set whether
 * this should be used.
 * 
 * Particle and exception parameter offsets are not implemented.  Their API entry points are retained for source
 * compatibility with OpenMM's NonbondedForce shape, but they throw an exception when called.
 */

class OPENMM_EXPORT_PERTURBNONBONDED PerturbNonbondedForce : public OpenMM::Force {
public:
    /**
     * This is an enumeration of the different methods that may be used for handling long range nonbonded forces.
     */
    enum PerturbNonbondedMethod {
        /**
         * No cutoff is applied to nonbonded interactions.  The full set of N^2 interactions is computed exactly.
         * This necessarily means that periodic boundary conditions cannot be used.  This is the default.
         */
        NoCutoff = 0,
        /**
         * Interactions beyond the cutoff distance are ignored.  Coulomb interactions closer than the cutoff distance
         * are modified using the reaction field method.
         */
        CutoffNonPeriodic = 1,
        /**
         * Periodic boundary conditions are used, so that each particle interacts only with the nearest periodic copy of
         * each other particle.  Interactions beyond the cutoff distance are ignored.  Coulomb interactions closer than the
         * cutoff distance are modified using the reaction field method.
         */
        CutoffPeriodic = 2,
        /**
         * Periodic boundary conditions are used, and Ewald summation is used to compute the Coulomb interaction of each particle
         * with all periodic copies of every other particle.
         */
        Ewald = 3,
        /**
         * Periodic boundary conditions are used, and Particle-Mesh Ewald (PME) summation is used to compute the Coulomb interaction of each particle
         * with all periodic copies of every other particle.
         */
        PME = 4,
        /**
         * Periodic boundary conditions are used, and Particle-Mesh Ewald (PME) summation is used to compute the interaction of each particle
         * with all periodic copies of every other particle for both Coulomb and Lennard-Jones.  No switching is used for either interaction.
         */
        LJPME = 5
    };
    /**
     * Create a PerturbNonbondedForce.
     */
    PerturbNonbondedForce();
    /**
     * Get the number of particles for which force field parameters have been defined.
     */
    int getNumParticles() const;
    /**
     * Get the number of special interactions that should be calculated differently from other interactions.
     */
    int getNumExceptions() const;
    /**
     * Get the number of global parameters that have been added.
     */
    int getNumGlobalParameters() const;
    /**
     * Get the number of particles parameter offsets that have been added.
     */
    int getNumParticleParameterOffsets() const;
    /**
     * Get the number of exception parameter offsets that have been added.
     */
    int getNumExceptionParameterOffsets() const;
    /**
     * Get the method used for handling long range nonbonded interactions.
     */
    PerturbNonbondedMethod getNonbondedMethod() const;
    /**
     * Set the method used for handling long range nonbonded interactions.
     */
    void setNonbondedMethod(PerturbNonbondedMethod method);
    /**
     * Get the cutoff distance (in nm) being used for nonbonded interactions.  If the PerturbNonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @return the cutoff distance, measured in nm
     */
    double getCutoffDistance() const;
    /**
     * Set the cutoff distance (in nm) being used for nonbonded interactions.  If the PerturbNonbondedMethod in use
     * is NoCutoff, this value will have no effect.
     *
     * @param distance    the cutoff distance, measured in nm
     */
    void setCutoffDistance(double distance);
    /**
     * Get whether a switching function is applied to the Lennard-Jones interaction.  If the nonbonded method is set
     * to NoCutoff, this option is ignored.
     */
    bool getUseSwitchingFunction() const;
    /**
     * Set whether a switching function is applied to the Lennard-Jones interaction.  If the nonbonded method is set
     * to NoCutoff, this option is ignored.
     */
    void setUseSwitchingFunction(bool use);
    /**
     * Get the distance at which the switching function begins to reduce the Lennard-Jones interaction.  This must be
     * less than the cutoff distance.
     */
    double getSwitchingDistance() const;
    /**
     * Set the distance at which the switching function begins to reduce the Lennard-Jones interaction.  This must be
     * less than the cutoff distance.
     */
    void setSwitchingDistance(double distance);
    /**
     * Get the dielectric constant to use for the solvent in the reaction field approximation.
     */
    double getReactionFieldDielectric() const;
    /**
     * Set the dielectric constant to use for the solvent in the reaction field approximation.
     */
    void setReactionFieldDielectric(double dielectric);
    /**
     * Get the error tolerance for Ewald summation.  This corresponds to the fractional error in the forces
     * which is acceptable.  This value is used to select the reciprocal space cutoff and separation
     * parameter so that the average error level will be less than the tolerance.  There is not a
     * rigorous guarantee that all forces on all atoms will be less than the tolerance, however.
     *
     * For PME calculations, if setPMEParameters() is used to set alpha to something other than 0,
     * this value is ignored.
     */
    double getEwaldErrorTolerance() const;
    /**
     * Set the error tolerance for Ewald summation.  This corresponds to the fractional error in the forces
     * which is acceptable.  This value is used to select the reciprocal space cutoff and separation
     * parameter so that the average error level will be less than the tolerance.  There is not a
     * rigorous guarantee that all forces on all atoms will be less than the tolerance, however.
     *
     * For PME calculations, if setPMEParameters() is used to set alpha to something other than 0,
     * this value is ignored.
     */
    void setEwaldErrorTolerance(double tol);
    /**
     * Get the parameters to use for PME calculations.  If alpha is 0 (the default), these parameters are
     * ignored and instead their values are chosen based on the Ewald error tolerance.
     *
     * @param[out] alpha   the separation parameter
     * @param[out] nx      the number of grid points along the X axis
     * @param[out] ny      the number of grid points along the Y axis
     * @param[out] nz      the number of grid points along the Z axis
     */
    void getPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Get the parameters to use for dispersion term in LJ-PME calculations.  If alpha is 0 (the default),
     * these parameters are ignored and instead their values are chosen based on the Ewald error tolerance.
     *
     * @param[out] alpha   the separation parameter
     * @param[out] nx      the number of dispersion grid points along the X axis
     * @param[out] ny      the number of dispersion grid points along the Y axis
     * @param[out] nz      the number of dispersion grid points along the Z axis
     */
    void getLJPMEParameters(double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Set the parameters to use for PME calculations.  If alpha is 0 (the default), these parameters are
     * ignored and instead their values are chosen based on the Ewald error tolerance.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void setPMEParameters(double alpha, int nx, int ny, int nz);
    /**
     * Set the parameters to use for the dispersion term in LJPME calculations.  If alpha is 0 (the default),
     * these parameters are ignored and instead their values are chosen based on the Ewald error tolerance.
     *
     * @param alpha   the separation parameter
     * @param nx      the number of grid points along the X axis
     * @param ny      the number of grid points along the Y axis
     * @param nz      the number of grid points along the Z axis
     */
    void setLJPMEParameters(double alpha, int nx, int ny, int nz);
    /**
     * Get the parameters being used for PME in a particular Context.  Because some platforms have restrictions
     * on the allowed grid sizes, the values that are actually used may be slightly different from those
     * specified with setPMEParameters(), or the standard values calculated based on the Ewald error tolerance.
     * See the manual for details.
     *
     * @param context      the Context for which to get the parameters
     * @param[out] alpha   the separation parameter
     * @param[out] nx      the number of grid points along the X axis
     * @param[out] ny      the number of grid points along the Y axis
     * @param[out] nz      the number of grid points along the Z axis
     */
    void getPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Get the PME parameters being used for the dispersion term for LJPME in a particular Context.  Because some
     * platforms have restrictions on the allowed grid sizes, the values that are actually used may be slightly different
     * from those specified with setPMEParameters(), or the standard values calculated based on the Ewald error tolerance.
     * See the manual for details.
     *
     * @param context      the Context for which to get the parameters
     * @param[out] alpha   the separation parameter
     * @param[out] nx      the number of grid points along the X axis
     * @param[out] ny      the number of grid points along the Y axis
     * @param[out] nz      the number of grid points along the Z axis
     */
    void getLJPMEParametersInContext(const Context& context, double& alpha, int& nx, int& ny, int& nz) const;
    /**
     * Add the nonbonded force parameters for a particle.  This should be called once for each particle
     * in the System.  When it is called for the i'th time, it specifies the parameters for the i'th particle.
     * For calculating the Lennard-Jones interaction between two particles, the arithmetic mean of the sigmas
     * and the geometric mean of the epsilons for the two interacting particles is used (the Lorentz-Berthelot
     * combining rule).
     *
     * @param charge    the charge of the particle, measured in units of the proton charge
     * @param sigma     the sigma parameter of the Lennard-Jones potential (corresponding to the van der Waals radius of the particle), measured in nm
     * @param epsilon   the epsilon parameter of the Lennard-Jones potential (corresponding to the well depth of the van der Waals interaction), measured in kJ/mol
     * @return the index of the particle that was added
     */
    int addParticle(double chargeA, double sigmaA, double epsilonA, double chargeB, double sigmaB, double epsilonB, bool isDummy, bool isRest2);
    /**
     * Get the nonbonded force parameters for a particle.
     *
     * @param index          the index of the particle for which to get parameters
     * @param[out] charge    the charge of the particle, measured in units of the proton charge
     * @param[out] sigma     the sigma parameter of the Lennard-Jones potential (corresponding to the van der Waals radius of the particle), measured in nm
     * @param[out] epsilon   the epsilon parameter of the Lennard-Jones potential (corresponding to the well depth of the van der Waals interaction), measured in kJ/mol
     */
    void getParticleParameters(int index, double& chargeA, double& sigmaA, double& epsilonA, double& chargeB, double& sigmaB, double& epsilonB, bool& isDummy, bool& isRest2) const;
    /**
     * Set the nonbonded force parameters for a particle.  When calculating the Lennard-Jones interaction between two particles,
     * it uses the arithmetic mean of the sigmas and the geometric mean of the epsilons for the two interacting particles
     * (the Lorentz-Berthelot combining rule).
     *
     * @param index     the index of the particle for which to set parameters
     * @param charge    the charge of the particle, measured in units of the proton charge
     * @param sigma     the sigma parameter of the Lennard-Jones potential (corresponding to the van der Waals radius of the particle), measured in nm
     * @param epsilon   the epsilon parameter of the Lennard-Jones potential (corresponding to the well depth of the van der Waals interaction), measured in kJ/mol
     */
    void setParticleParameters(int index, double chargeA, double sigmaA, double epsilonA, double chargeB, double sigmaB, double epsilonB, bool isDummy, bool isRest2);
    /**
     * Add an interaction to the list of exceptions that should be calculated differently from other interactions.
     * If chargeProd and epsilon are both equal to 0, this will cause the interaction to be completely omitted from
     * force and energy calculations.
     * 
     * Regardless of the PerturbNonbondedMethod used by this Force, cutoffs are never applied to exceptions.  That is because
     * they are primarily used for 1-4 interactions, which are really a type of bonded interaction and are parametrized
     * together with the other bonded interactions.
     *
     * In many cases, you can use createExceptionsFromBonds() rather than adding each exception explicitly.
     *
     * @param particle1  the index of the first particle involved in the interaction
     * @param particle2  the index of the second particle involved in the interaction
     * @param chargeProd the scaled product of the atomic charges (i.e. the strength of the Coulomb interaction), measured in units of the proton charge squared
     * @param sigma      the sigma parameter of the Lennard-Jones potential (corresponding to the van der Waals radius of the particle), measured in nm
     * @param epsilon    the epsilon parameter of the Lennard-Jones potential (corresponding to the well depth of the van der Waals interaction), measured in kJ/mol
     * @param replace    determines the behavior if there is already an exception for the same two particles.  If true, the existing one is replaced.  If false,
     *                   an exception is thrown.
     * @return the index of the exception that was added
     */
    int addException(int particle1, int particle2, double chargeProd, double sigma, double epsilon, bool replace = false);
    /**
     * Get the force field parameters for an interaction that should be calculated differently from others.
     *
     * @param index           the index of the interaction for which to get parameters
     * @param[out] particle1  the index of the first particle involved in the interaction
     * @param[out] particle2  the index of the second particle involved in the interaction
     * @param[out] chargeProd the scaled product of the atomic charges (i.e. the strength of the Coulomb interaction), measured in units of the proton charge squared
     * @param[out] sigma      the sigma parameter of the Lennard-Jones potential (corresponding to the van der Waals radius of the particle), measured in nm
     * @param[out] epsilon    the epsilon parameter of the Lennard-Jones potential (corresponding to the well depth of the van der Waals interaction), measured in kJ/mol
     */
    void getExceptionParameters(int index, int& particle1, int& particle2, double& chargeProd, double& sigma, double& epsilon) const;
    /**
     * Set the force field parameters for an interaction that should be calculated differently from others.
     * If chargeProd and epsilon are both equal to 0, this will cause the interaction to be completely omitted from
     * force and energy calculations.
     * 
     * Regardless of the PerturbNonbondedMethod used by this Force, cutoffs are never applied to exceptions.  That is because
     * they are primarily used for 1-4 interactions, which are really a type of bonded interaction and are parametrized
     * together with the other bonded interactions.
     *
     * @param index      the index of the interaction for which to get parameters
     * @param particle1  the index of the first particle involved in the interaction
     * @param particle2  the index of the second particle involved in the interaction
     * @param chargeProd the scaled product of the atomic charges (i.e. the strength of the Coulomb interaction), measured in units of the proton charge squared
     * @param sigma      the sigma parameter of the Lennard-Jones potential (corresponding to the van der Waals radius of the particle), measured in nm
     * @param epsilon    the epsilon parameter of the Lennard-Jones potential (corresponding to the well depth of the van der Waals interaction), measured in kJ/mol
     */
    void setExceptionParameters(int index, int particle1, int particle2, double chargeProd, double sigma, double epsilon);
    /**
     * Identify exceptions based on the molecular topology.  Particles which are separated by one or two bonds are set
     * to not interact at all, while pairs of particles separated by three bonds (known as "1-4 interactions") have
     * their Coulomb and Lennard-Jones interactions reduced by a fixed factor.
     *
     * @param bonds           the set of bonds based on which to construct exceptions.  Each element specifies the indices of
     *                        two particles that are bonded to each other.
     * @param coulomb14Scale  pairs of particles separated by three bonds will have the strength of their Coulomb interaction
     *                        multiplied by this factor
     * @param lj14Scale       pairs of particles separated by three bonds will have the strength of their Lennard-Jones interaction
     *                        multiplied by this factor
     */
    void createExceptionsFromBonds(const std::vector<std::pair<int, int> >& bonds, double coulomb14Scale, double lj14Scale);
    /**
     * Add a new global parameter that parameter offsets may depend on.  The default value provided to
     * this method is the initial value of the parameter in newly created Contexts.  You can change
     * the value at any time by calling setParameter() on the Context.
     * 
     * @param name             the name of the parameter
     * @param defaultValue     the default value of the parameter
     * @return the index of the parameter that was added
     */
    int addGlobalParameter(const std::string& name, double defaultValue);
    /**
     * Get the name of a global parameter.
     *
     * @param index     the index of the parameter for which to get the name
     * @return the parameter name
     */
    const std::string& getGlobalParameterName(int index) const;
    /**
     * Set the name of a global parameter.
     *
     * @param index          the index of the parameter for which to set the name
     * @param name           the name of the parameter
     */
    void setGlobalParameterName(int index, const std::string& name);
    /**
     * Get the default value of a global parameter.
     *
     * @param index     the index of the parameter for which to get the default value
     * @return the parameter default value
     */
    double getGlobalParameterDefaultValue(int index) const;
    /**
     * Set the default value of a global parameter.
     *
     * @param index          the index of the parameter for which to set the default value
     * @param defaultValue   the default value of the parameter
     */
    void setGlobalParameterDefaultValue(int index, double defaultValue);
    /**
     * Add an offset to the per-particle parameters of a particular particle, based on a global parameter.
     * 
     * @param parameter       the name of the global parameter.  It must have already been added with addGlobalParameter().
     *                        Its value can be modified at any time by calling Context::setParameter().
     * @param particleIndex   the index of the particle whose parameters are affected
     * @param chargeScale     this value multiplied by the parameter value is added to the particle's charge
     * @param sigmaScale      this value multiplied by the parameter value is added to the particle's sigma
     * @param epsilonScale    this value multiplied by the parameter value is added to the particle's epsilon
     * @return the index of the offset that was added
     */
    int addParticleParameterOffset(const std::string& parameter, int particleIndex, double chargeScale, double sigmaScale, double epsilonScale);
    /**
     * Get the offset added to the per-particle parameters of a particular particle, based on a global parameter.
     * 
     * @param index           the index of the offset to query, as returned by addParticleParameterOffset()
     * @param parameter       the name of the global parameter
     * @param particleIndex   the index of the particle whose parameters are affected
     * @param chargeScale     this value multiplied by the parameter value is added to the particle's charge
     * @param sigmaScale      this value multiplied by the parameter value is added to the particle's sigma
     * @param epsilonScale    this value multiplied by the parameter value is added to the particle's epsilon
     */
    void getParticleParameterOffset(int index, std::string& parameter, int& particleIndex, double& chargeScale, double& sigmaScale, double& epsilonScale) const;
    /**
     * Set the offset added to the per-particle parameters of a particular particle, based on a global parameter.
     * 
     * @param index           the index of the offset to modify, as returned by addParticleParameterOffset()
     * @param parameter       the name of the global parameter.  It must have already been added with addGlobalParameter().
     *                        Its value can be modified at any time by calling Context::setParameter().
     * @param particleIndex   the index of the particle whose parameters are affected
     * @param chargeScale     this value multiplied by the parameter value is added to the particle's charge
     * @param sigmaScale      this value multiplied by the parameter value is added to the particle's sigma
     * @param epsilonScale    this value multiplied by the parameter value is added to the particle's epsilon
     */
    void setParticleParameterOffset(int index, const std::string& parameter, int particleIndex, double chargeScale, double sigmaScale, double epsilonScale);
    /**
     * Add an offset to the parameters of a particular exception, based on a global parameter.
     * 
     * @param parameter       the name of the global parameter.  It must have already been added with addGlobalParameter().
     *                        Its value can be modified at any time by calling Context::setParameter().
     * @param exceptionIndex  the index of the exception whose parameters are affected
     * @param chargeProdScale this value multiplied by the parameter value is added to the exception's charge product
     * @param sigmaScale      this value multiplied by the parameter value is added to the exception's sigma
     * @param epsilonScale    this value multiplied by the parameter value is added to the exception's epsilon
     * @return the index of the offset that was added
     */
    int addExceptionParameterOffset(const std::string& parameter, int exceptionIndex, double chargeProdScale, double sigmaScale, double epsilonScale);
    /**
     * Get the offset added to the parameters of a particular exception, based on a global parameter.
     * 
     * @param index           the index of the offset to query, as returned by addExceptionParameterOffset()
     * @param parameter       the name of the global parameter
     * @param exceptionIndex  the index of the exception whose parameters are affected
     * @param chargeProdScale this value multiplied by the parameter value is added to the exception's charge product
     * @param sigmaScale      this value multiplied by the parameter value is added to the exception's sigma
     * @param epsilonScale    this value multiplied by the parameter value is added to the exception's epsilon
     */
    void getExceptionParameterOffset(int index, std::string& parameter, int& exceptionIndex, double& chargeProdScale, double& sigmaScale, double& epsilonScale) const;
    /**
     * Set the offset added to the parameters of a particular exception, based on a global parameter.
     * 
     * @param index           the index of the offset to modify, as returned by addExceptionParameterOffset()
     * @param parameter       the name of the global parameter.  It must have already been added with addGlobalParameter().
     *                        Its value can be modified at any time by calling Context::setParameter().
     * @param exceptionIndex  the index of the exception whose parameters are affected
     * @param chargeProdScale this value multiplied by the parameter value is added to the exception's charge product
     * @param sigmaScale      this value multiplied by the parameter value is added to the exception's sigma
     * @param epsilonScale    this value multiplied by the parameter value is added to the exception's epsilon
     */
    void setExceptionParameterOffset(int index, const std::string& parameter, int exceptionIndex, double chargeProdScale, double sigmaScale, double epsilonScale);
    /**
     * Get whether to add a contribution to the energy that approximately represents the effect of Lennard-Jones
     * interactions beyond the cutoff distance.  The energy depends on the volume of the periodic box, and is only
     * applicable when periodic boundary conditions are used.  When running simulations at constant pressure, adding
     * this contribution can improve the quality of results.
     */
    bool getUseDispersionCorrection() const {
        return useDispersionCorrection;
    }
    /**
     * Set whether to add a contribution to the energy that approximately represents the effect of Lennard-Jones
     * interactions beyond the cutoff distance.  The energy depends on the volume of the periodic box, and is only
     * applicable when periodic boundary conditions are used.  When running simulations at constant pressure, adding
     * this contribution can improve the quality of results.
     */
    void setUseDispersionCorrection(bool useCorrection) {
        useDispersionCorrection = useCorrection;
    }
    /**
     * Get the force group that reciprocal space interactions for Ewald or PME are included in.  This allows multiple
     * time step integrators to evaluate direct and reciprocal space interactions at different intervals: getForceGroup()
     * specifies the group for direct space, and getReciprocalSpaceForceGroup() specifies the group for reciprocal space.
     * If this is -1 (the default value), the same force group is used for reciprocal space as for direct space.
     */
    int getReciprocalSpaceForceGroup() const;
    /**
     * Set the force group that reciprocal space interactions for Ewald or PME are included in.  This allows multiple
     * time step integrators to evaluate direct and reciprocal space interactions at different intervals: setForceGroup()
     * specifies the group for direct space, and setReciprocalSpaceForceGroup() specifies the group for reciprocal space.
     * If this is -1 (the default value), the same force group is used for reciprocal space as for direct space.
     *
     * @param group    the group index.  Legal values are between 0 and 31 (inclusive), or -1 to use the same force group
     *                 that is specified for direct space.
     */
    void setReciprocalSpaceForceGroup(int group);
    /**
     * Get whether to include direct space interactions when calculating forces and energies.  This is useful if you want
     * to completely replace the direct space calculation, typically with a CustomPerturbNonbondedForce that computes it in a
     * nonstandard way, while still using this object for the reciprocal space calculation.
     */
    bool getIncludeDirectSpace() const;
    /**
     * Set whether to include direct space interactions when calculating forces and energies.  This is useful if you want
     * to completely replace the direct space calculation, typically with a CustomPerturbNonbondedForce that computes it in a
     * nonstandard way, while still using this object for the reciprocal space calculation.
     */
    void setIncludeDirectSpace(bool include);
    /**
     * Update the particle and exception parameters in a Context to match those stored in this Force object.  This method
     * provides an efficient method to update certain parameters in an existing Context without needing to reinitialize it.
     * Simply call setParticleParameters() and setExceptionParameters() to modify this object's parameters, then call
     * updateParametersInContext() to copy them over to the Context.
     *
     * This method has several limitations.  The only information it updates is the parameters of particles and exceptions,
     * as well as parameter offsets for particles and exceptions.  All other aspects of the Force (the nonbonded method,
     * the cutoff distance, etc.) are unaffected and can only be changed by reinitializing the Context.  Furthermore,
     * only the chargeProd, sigma, and epsilon values of an exception can be changed; the pair of particles involved in the
     * exception cannot change.  Likewise, it can update charge, sigma and epsilon for a parameter offset, but not the
     * identify of the particle or exception the offset is applied to, or which global parameter it is based on.  Finally,
     * this method cannot be used to add new particles or exceptions, only to change the parameters of existing ones.
     */
    void updateParametersInContext(Context& context);
    /**
     * Returns whether or not this force makes use of periodic boundary
     * conditions.
     *
     * @returns true if force uses PBC and false otherwise
     */
    bool usesPeriodicBoundaryConditions() const {
        return nonbondedMethod == PerturbNonbondedForce::CutoffPeriodic ||
               nonbondedMethod == PerturbNonbondedForce::Ewald ||
               nonbondedMethod == PerturbNonbondedForce::PME ||
               nonbondedMethod == PerturbNonbondedForce::LJPME;
    }
    /**
     * Get whether periodic boundary conditions should be applied to exceptions.  Usually this is not
     * appropriate, because exceptions are normally used to represent bonded interactions (1-2, 1-3, and
     * 1-4 pairs), but there are situations when it does make sense.  For nonbonded, you may want to simulate
     * an infinite chain where one end of a molecule is bonded to the opposite end of the next periodic
     * copy.
     * 
     * Regardless of this value, periodic boundary conditions are only applied to exceptions if they also
     * are applied to other interactions.  If the nonbonded method is NoCutoff or CutoffNonPeriodic, this
     * value is ignored.  Also note that cutoffs are never applied to exceptions, again because they are
     * normally used to represent bonded interactions.
     */
    bool getExceptionsUsePeriodicBoundaryConditions() const;
    /**
     * Set whether periodic boundary conditions should be applied to exceptions.  Usually this is not
     * appropriate, because exceptions are normally used to represent bonded interactions (1-2, 1-3, and
     * 1-4 pairs), but there are situations when it does make sense.  For nonbonded, you may want to simulate
     * an infinite chain where one end of a molecule is bonded to the opposite end of the next periodic
     * copy.
     * 
     * Regardless of this value, periodic boundary conditions are only applied to exceptions if they also
     * get applied to other interactions.  If the nonbonded method is NoCutoff or CutoffNonPeriodic, this
     * value is ignored.  Also note that cutoffs are never applied to exceptions, again because they are
     * normally used to represent bonded interactions.
     */
    void setExceptionsUsePeriodicBoundaryConditions(bool periodic);

    void setSoftcoreAlpha(double alpha);
    double getSoftcoreAlpha() const;

    void setLJForceGroup(int ljForceGroup);
    int getLJForceGroup() const;

    void setDirectSpaceForceGroup(int directSpaceForceGroup);
    int getDirectSpaceForceGroup() const;

    void setIncludeLJ(bool includeLJ);
    bool getIncludeLJ() const;

    void enableEnergyConstantCache(bool enable);
    bool getEnableEnergyConstantCache() const;
    void setLambdaSchedules(
        const std::vector<double>& lambdaA_charge_schedule, 
        const std::vector<double>& lambdaB_charge_schedule,
        const std::vector<double>& lambdaC_charge_schedule,
        const std::vector<double>& lambdaA_vdw_schedule,
        const std::vector<double>& lambdaB_vdw_schedule,
        const std::vector<double>& rest2_scale_schedule);

    map<string, vector<double>> getLambdaSchedules() const;
    int getNumReplicas() const;

    /**
     * Evaluate reaction-field/direct interaction energies for real system particles.
     * All pointer arguments refer to CUDA arrays owned by the caller.  Positions and
     * parameters are packed float3 arrays; parameters contain charge, half-sigma,
     * and twice-sqrt-epsilon.  Energies are accumulated into the supplied float array.
     */
    void getInteractionEnergyForRealParticlesCuda(Context& context, int numRealAtoms,
            CudaArrayDevicePointer positions, CudaArrayDevicePointer parameters,
            CudaArrayDevicePointer excludedAtomStarts, CudaArrayDevicePointer excludedAtomCounts,
            CudaArrayDevicePointer excludedAtoms, CudaArrayDevicePointer forces,
            CudaArrayDevicePointer energies);

    /** Evaluate PME direct-space and Lennard-Jones energies for candidate particles. */
    void getDirectEnergyForCandidateParticlesCuda(Context& context, int numCandidateAtoms,
            CudaArrayDevicePointer positions, CudaArrayDevicePointer parameters,
            CudaArrayDevicePointer forces, CudaArrayDevicePointer energies);

    /**
     * Evaluate PME direct-space and Lennard-Jones energies using positions in the
     * Context's native precision layout.  The base array is float4 in single and
     * mixed precision and double4 in double precision.  The correction array must
     * be a float4 array in mixed precision and zero in single or double precision.
     */
    void getDirectEnergyForCandidateParticlesCudaContextPrecision(Context& context,
            int numCandidateAtoms, CudaArrayDevicePointer positionBase,
            CudaArrayDevicePointer positionCorrection, CudaArrayDevicePointer parameters,
            CudaArrayDevicePointer forces, CudaArrayDevicePointer energies);

    /** Interpolate the current PME reciprocal potential at candidate positions. */
    void getPMEPotentialAtPositionsCuda(Context& context, int numPositions,
            CudaArrayDevicePointer positions, CudaArrayDevicePointer potentials);

    /**
     * Interpolate the current PME reciprocal potential using positions in the
     * Context's native precision layout.  Position base and correction arrays use
     * the same layout contract as getDirectEnergyForCandidateParticlesCudaContextPrecision().
     */
    void getPMEPotentialAtPositionsCudaContextPrecision(Context& context, int numPositions,
            CudaArrayDevicePointer positionBase, CudaArrayDevicePointer positionCorrection,
            CudaArrayDevicePointer potentials);

    /**
     * Decompose the current PME mesh reciprocal energy symmetrically over all
     * real atoms as 0.5*sqrt(ONE_4PI_EPS0)*q_i*phi_full(i).  Output is in host
     * atom order and uses the CUDA mixed scalar type (float for single,
     * double for mixed/double).  These values sum to the mesh reciprocal
     * energy but are not single-particle removal energies.
     */
    void getSymmetricPMEReciprocalAtomicEnergiesCuda(Context& context,
            CudaArrayDevicePointer energies, CudaArrayDevicePointer sum);

    /**
     * Build a dedicated PME scratch grid from caller-owned source particles and
     * interpolate its raw reciprocal potential at caller-owned query positions.
     * Positions are float4 and charges are float; the main Context PME grids are
     * not modified.  All spread/FFT/convolution/gather work remains on CUDA.
     */
    void getPMEPotentialFromParticlesCuda(Context& context, int numSourceParticles,
            CudaArrayDevicePointer sourcePositions, CudaArrayDevicePointer sourceCharges,
            int numQueryPositions, CudaArrayDevicePointer queryPositions,
            CudaArrayDevicePointer queryPotentials);

    /**
     * Build and query a scratch PME grid using source and query positions in the
     * Context's native precision layout.  Each base/correction pair follows the
     * ContextPrecision position contract described above.
     */
    void getPMEPotentialFromParticlesCudaContextPrecision(Context& context,
            int numSourceParticles, CudaArrayDevicePointer sourcePositionBase,
            CudaArrayDevicePointer sourcePositionCorrection, CudaArrayDevicePointer sourceCharges,
            int numQueryPositions, CudaArrayDevicePointer queryPositionBase,
            CudaArrayDevicePointer queryPositionCorrection, CudaArrayDevicePointer queryPotentials);

    /**
     * Compute one reciprocal mesh self energy for each fixed-size candidate.
     * Candidates are evaluated sequentially through a reusable scratch PME grid,
     * so no cross-candidate terms can enter the result.  Results remain on CUDA.
     */
    void getPMECandidateSelfEnergiesCuda(Context& context, int numCandidates,
            int numAtomsPerCandidate, CudaArrayDevicePointer candidatePositions,
            CudaArrayDevicePointer candidateCharges, CudaArrayDevicePointer energies);

    /**
     * Compute candidate reciprocal mesh self energies using positions in the
     * Context's native precision layout.
     */
    void getPMECandidateSelfEnergiesCudaContextPrecision(Context& context, int numCandidates,
            int numAtomsPerCandidate, CudaArrayDevicePointer candidatePositionBase,
            CudaArrayDevicePointer candidatePositionCorrection,
            CudaArrayDevicePointer candidateCharges, CudaArrayDevicePointer energies);

    /**
     * Evaluate PME/Ewald direct-space and LJ interactions between candidate atoms
     * and a caller-owned subset of real system atoms.  The source atom indices are
     * host-order indices; outputs are per candidate atom and remain on CUDA.
     */
    void getDirectEnergyForCandidateSubsetCuda(Context& context, int numCandidateAtoms,
            CudaArrayDevicePointer candidatePositions, CudaArrayDevicePointer candidateParameters,
            int numSourceAtoms, CudaArrayDevicePointer sourceAtomIndices,
            CudaArrayDevicePointer energies);

    /**
     * Evaluate candidate-to-subset direct interactions using candidate positions
     * in the Context's native precision layout.
     */
    void getDirectEnergyForCandidateSubsetCudaContextPrecision(Context& context,
            int numCandidateAtoms, CudaArrayDevicePointer candidatePositionBase,
            CudaArrayDevicePointer candidatePositionCorrection,
            CudaArrayDevicePointer candidateParameters, int numSourceAtoms,
            CudaArrayDevicePointer sourceAtomIndices, CudaArrayDevicePointer energies);

    /**
     * Evaluate PME direct-space/LJ energy for real particles at their current
     * positions, excluding the query molecule itself.  Inputs use host-order
     * atom indices; outputs are per query atom in molecule-major order.
     */
    void getDirectEnergyForRealParticlesPmeCuda(Context& context, int numRealAtoms,
            CudaArrayDevicePointer realAtomIndices, int numAtomsPerMolecule,
            CudaArrayDevicePointer energies);

    /**
     * Return platform-specific Context data for cooperating CUDA plugins and tests.
     * This experimental CUDA integration hook is not part of the portable Force API.
     */
    void* getCudaPlatformData(Context& context) const;

    /** Testing-only experimental hook for comparing pruned and exhaustive-block CUDA queries. */
    void setCudaParticleQueryBlockPruningForTesting(Context& context, bool enabled);

protected:
    ForceImpl* createImpl() const;
private:
    class ParticleInfo;
    class ExceptionInfo;
    class GlobalParameterInfo;
    class ParticleOffsetInfo;
    class ExceptionOffsetInfo;
    PerturbNonbondedMethod nonbondedMethod;
    double cutoffDistance, switchingDistance, rfDielectric, ewaldErrorTol, alpha, dalpha;
    bool useSwitchingFunction, useDispersionCorrection, exceptionsUsePeriodic, includeDirectSpace;
    int recipForceGroup, nx, ny, nz, dnx, dny, dnz;

    double softcore_alpha = 1.0;
    int ljForceGroup = -1;
    int directSpaceForceGroup = -1;
    bool includeLJ = true;
    bool cacheEnergyConstants = false;

    int numReplicas = 0;
    map<string, vector<double>> lambdaSchedules;


    void addExclusionsToSet(const std::vector<std::set<int> >& bonded12, std::set<int>& exclusions, int baseParticle, int fromParticle, int currentLevel) const;
    int getGlobalParameterIndex(const std::string& parameter) const;
    std::vector<ParticleInfo> particles;
    std::vector<ExceptionInfo> exceptions;
    std::vector<GlobalParameterInfo> globalParameters;
    std::vector<ParticleOffsetInfo> particleOffsets;
    std::vector<ExceptionOffsetInfo> exceptionOffsets;
    std::map<std::pair<int, int>, int> exceptionMap;
    mutable int numContexts, firstChangedParticle, lastChangedParticle, firstChangedException, lastChangedException;
};

/**
 * This is an internal class used to record information about a particle.
 * @private
 */
class PerturbNonbondedForce::ParticleInfo {
public:
    double chargeA, sigmaA, epsilonA;
    double chargeB, sigmaB, epsilonB;
    bool isDummy, isRest2;
    ParticleInfo() {
        chargeA = sigmaA = epsilonA = 0.0;
    }
    ParticleInfo(double chargeA, double sigmaA, double epsilonA, double chargeB, double sigmaB, double epsilonB, bool isDummy, bool isRest2) :
        chargeA(chargeA), sigmaA(sigmaA), epsilonA(epsilonA), chargeB(chargeB), sigmaB(sigmaB), epsilonB(epsilonB), isDummy(isDummy), isRest2(isRest2) {
    }
};

/**
 * This is an internal class used to record information about an exception.
 * @private
 */
class PerturbNonbondedForce::ExceptionInfo {
public:
    int particle1, particle2;
    double chargeProd, sigma, epsilon;
    ExceptionInfo() {
        particle1 = particle2 = -1;
        chargeProd = sigma = epsilon = 0.0;
    }
    ExceptionInfo(int particle1, int particle2, double chargeProd, double sigma, double epsilon) :
        particle1(particle1), particle2(particle2), chargeProd(chargeProd), sigma(sigma), epsilon(epsilon) {
    }
};

/**
 * This is an internal class used to record information about a global parameter.
 * @private
 */
class PerturbNonbondedForce::GlobalParameterInfo {
public:
    std::string name;
    double defaultValue;
    GlobalParameterInfo() {
    }
    GlobalParameterInfo(const std::string& name, double defaultValue) : name(name), defaultValue(defaultValue) {
    }
};

/**
 * This is an internal class used to record information about a particle parameter offset.
 * @private
 */
class PerturbNonbondedForce::ParticleOffsetInfo {
public:
    int particle, parameter;
    double chargeScale, sigmaScale, epsilonScale;
    ParticleOffsetInfo() {
        particle = parameter = -1;
        chargeScale = sigmaScale = epsilonScale = 0.0;
    }
    ParticleOffsetInfo(int parameter, int particle, double chargeScale, double sigmaScale, double epsilonScale) :
        parameter(parameter), particle(particle), chargeScale(chargeScale), sigmaScale(sigmaScale), epsilonScale(epsilonScale) {
    }
};

/**
 * This is an internal class used to record information about an exception parameter offset.
 * @private
 */
class PerturbNonbondedForce::ExceptionOffsetInfo {
public:
    int exception, parameter;
    double chargeProdScale, sigmaScale, epsilonScale;
    ExceptionOffsetInfo() {
        exception = parameter = -1;
        chargeProdScale = sigmaScale = epsilonScale = 0.0;
    }
    ExceptionOffsetInfo(int parameter, int exception, double chargeProdScale, double sigmaScale, double epsilonScale) :
        parameter(parameter), exception(exception), chargeProdScale(chargeProdScale), sigmaScale(sigmaScale), epsilonScale(epsilonScale) {
    }
};

} // namespace PerturbNonbondedPlugin

#endif /*OPENMM_PERTURBNONBONDEDFORCE_H_*/
