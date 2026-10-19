#ifndef OPENMM_COMMONKERNELSOURCES_H_
#define OPENMM_COMMONKERNELSOURCES_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2019 Stanford University and the Authors.           *
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

#include "openmm/common/windowsExportCommon.h"
#include <string>

namespace OpenMM {

/**
 * This class is a central holding place for the source code of common kernels.
 * The CMake build script inserts declarations into it based on the .cc files in the
 * kernels subfolder.
 */

class OPENMM_EXPORT_COMMON CommonKernelSources {
public:
static const std::string andersenThermostat;
static const std::string angleForce;
static const std::string atmforce;
static const std::string bondForce;
static const std::string brownian;
static const std::string cmapTorsionForce;
static const std::string constantPotential;
static const std::string constantPotentialCGSolver;
static const std::string constantPotentialCoulombEnergyForces;
static const std::string constantPotentialExceptions;
static const std::string constantPotentialExclusions;
static const std::string constantPotentialMatrixSolver;
static const std::string constantPotentialSolver;
static const std::string constraints;
static const std::string copyCoordinateBuffers;
static const std::string coulombLennardJones;
static const std::string customCVForce;
static const std::string customCentroidBond;
static const std::string customCppForce;
static const std::string customExternalForce;
static const std::string customGBChainRule;
static const std::string customGBEnergyN2;
static const std::string customGBEnergyN2_cpu;
static const std::string customGBEnergyPerParticle;
static const std::string customGBGradientChainRule;
static const std::string customGBValueN2;
static const std::string customGBValueN2_cpu;
static const std::string customGBValuePerParticle;
static const std::string customHbondForce;
static const std::string customIntegrator;
static const std::string customIntegratorPerDof;
static const std::string customManyParticle;
static const std::string customNonbonded;
static const std::string customNonbondedComputedValues;
static const std::string customNonbondedGroups;
static const std::string dpd;
static const std::string ewald;
static const std::string gayBerne;
static const std::string gbsaObc;
static const std::string gbsaObc2;
static const std::string gbsaObcReductions;
static const std::string gbsaObc_cpu;
static const std::string harmonicAngleForce;
static const std::string harmonicBondForce;
static const std::string integrationUtilities;
static const std::string langevinMiddle;
static const std::string lcpo;
static const std::string minimize;
static const std::string monteCarloBarostat;
static const std::string nonbondedExceptions;
static const std::string nonbondedParameters;
static const std::string noseHooverChain;
static const std::string noseHooverIntegrator;
static const std::string orientationRestraintForce;
static const std::string periodicTorsionForce;
static const std::string pme;
static const std::string pmeExclusions;
static const std::string pointFunctions;
static const std::string qtb;
static const std::string rbTorsionForce;
static const std::string removeCM;
static const std::string rg;
static const std::string rmsd;
static const std::string torsionForce;
static const std::string verlet;

};

} // namespace OpenMM

#endif /*OPENMM_COMMONKERNELSOURCES_H_*/
