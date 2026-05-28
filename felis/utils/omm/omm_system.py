# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import math
from pathlib import Path
from typing import Optional

from openmm import BrownianIntegrator
from openmm import CMMotionRemover
from openmm import CustomBondForce
from openmm import CustomCompoundBondForce
from openmm import CustomExternalForce
from openmm import CustomNonbondedForce
from openmm import Force
from openmm import Integrator
from openmm import LangevinMiddleIntegrator
from openmm import MonteCarloBarostat
from openmm import NonbondedForce
from openmm import PeriodicTorsionForce
from openmm import Platform
from openmm import System
from openmm.app import GromacsGroFile
from openmm.app import GromacsTopFile
from openmm.app import PDBFile
from openmm.app import Simulation
from openmm.app import Topology
import openmm.app.forcefield as ff
import openmm.unit as unit

from bytemol.core import Molecule
from bytemol.core import MoleculeGraph
from bytemol.toolkit.system_builder.system_builder_tools import get_num_alchemwat
from felis.configs import GlobalKeys
from felis.configs import IntegratorNameOption
from felis.configs import MinimizeRelaxOption
from felis.utils.omm.fire2_minimizer import FIRE2Integrator

logger = logging.getLogger(__name__)

#######################
# initial information #
#######################


def _get_pdb(pdb: str) -> PDBFile:
    assert pdb.endswith(".pdb") and Path(pdb).exists()
    return PDBFile(pdb)


def _get_gro(gro: str) -> GromacsGroFile:
    assert gro.endswith(".gro") and Path(gro).exists()
    return GromacsGroFile(gro)


def get_initial_pos_pbc(gk: GlobalKeys):
    if gk.filename.crd.endswith(".pdb"):
        pdb = _get_pdb(gk.filename.crd)
        return pdb.positions, pdb.topology.getPeriodicBoxVectors()
    elif gk.filename.crd.endswith(".gro"):
        gro = _get_gro(gk.filename.crd)
        return gro.positions, gro.getPeriodicBoxVectors()
    else:
        raise NotImplementedError(f"cannot get pos and PBC from {gk.filename.crd})")


################
# simple edits #
################


def write_pdb(otop: Topology, pos, pbcvecs, newpdb: str) -> None:
    otop.setPeriodicBoxVectors(pbcvecs)
    PDBFile.writeFile(otop, pos, open(newpdb, "w"), keepIds=True)


def _set_unique_forcegroup(osys: System):
    for i, f in enumerate(osys.getForces()):
        f.setForceGroup(i + 1)


######################################
# other forces -- Position Restraint #
######################################


def _make_position_restraints(pos, index, k_kcal, tol_angstrom) -> CustomExternalForce:
    """
    0.5 sum k (r-r0)^2
    """
    fc = k_kcal * (unit.kilocalorie_per_mole / unit.angstrom**2) / (unit.kilojoule_per_mole / unit.nanometer**2)
    tol = tol_angstrom * unit.angstrom / unit.nanometer
    expression = "0.5*fc*select(step(dist-tol), (dist-tol)^2, 0); dist=periodicdistance(x,y,z,x0,y0,z0)"
    posres = CustomExternalForce(expression)
    posres.addPerParticleParameter("x0")
    posres.addPerParticleParameter("y0")
    posres.addPerParticleParameter("z0")
    posres.addPerParticleParameter("fc")
    posres.addPerParticleParameter("tol")
    pos2 = pos.value_in_unit(unit.nanometer)
    for p in index:
        x0 = pos2[p][0]
        y0 = pos2[p][1]
        z0 = pos2[p][2]
        posres.addParticle(p, [x0, y0, z0, fc, tol])
    return posres


def _add_position_restraints(osys: System, gk: GlobalKeys):
    posres_atoms = gk.posres.atoms
    if posres_atoms is None:
        return
    initpos, _ = get_initial_pos_pbc(gk)
    k_kcal = gk.posres.k_kcal
    if gk.ab.ligatoms is not None:
        k_kcal *= gk.ab.reslam
        logger.info(f"Lambda for restraints {gk.ab.reslam}")
    tol_angstrom = gk.posres.tol_angstrom
    posres = _make_position_restraints(initpos, posres_atoms, k_kcal, tol_angstrom)
    osys.addForce(posres)
    logger.info(f"Position Restraints: {k_kcal} kcal {tol_angstrom} A atoms {posres_atoms}")


###########################
# other forces -- Boresch #
###########################


def _make_boresch_restraints(lig3, pro3, kr_kcal, kangle_kcal, kdih_kcal, r0, theta0, phi0, alpha0, beta0,
                             gamma0) -> CustomCompoundBondForce:
    radian = 180 / math.pi
    kr = kr_kcal * (unit.kilocalories_per_mole / unit.angstrom**2) / (unit.kilojoules_per_mole / unit.nanometer**2)
    ka = kangle_kcal * (unit.kilocalories_per_mole / unit.radian**2) / (unit.kilojoules_per_mole / unit.radian**2)
    kd = kdih_kcal * unit.kilocalories_per_mole / unit.kilojoules_per_mole

    e_translate = "0.5*kr*(r-r0)^2 + 0.5*ka*(theta-theta0)^2 + kd*(1-cos(phi-phi0)) + "
    e_rotate = "0.5*ka*(alpha-alpha0)^2 + kd*(1-cos(beta-beta0)) + kd*(1-cos(gamma-gamma0));"
    # p1/2/3, p4/5/6 = Lig1/2/3, Pro1/2/3
    # r = L1,P1, theta=L1,P1,P2, phi=L1,P1,P2,P3
    #     p1,p4        p1,p4,p5      p1,p4,p5,p6
    # alpha=P1,L1,L2, beta=P2,P1,L1,L2, gamma=P1,L1,L2,L3
    #       p4,p1,p2       p5,p4,p1,p2        p4,p1,p2,p3
    e_def1 = "r=distance(p1,p4); theta=angle(p1,p4,p5); phi=dihedral(p1,p4,p5,p6);"
    e_def2 = "alpha=angle(p4,p1,p2); beta=dihedral(p5,p4,p1,p2); gamma=dihedral(p4,p1,p2,p3)"
    energy = e_translate + e_rotate + e_def1 + e_def2
    boresch = CustomCompoundBondForce(6, energy)
    boresch.setUsesPeriodicBoundaryConditions(True)
    boresch.addPerBondParameter("kr")
    boresch.addPerBondParameter("ka")
    boresch.addPerBondParameter("kd")
    boresch.addPerBondParameter("r0")
    boresch.addPerBondParameter("theta0")
    boresch.addPerBondParameter("phi0")
    boresch.addPerBondParameter("alpha0")
    boresch.addPerBondParameter("beta0")
    boresch.addPerBondParameter("gamma0")
    boresch.addBond([lig3[0], lig3[1], lig3[2], pro3[0], pro3[1], pro3[2]],
                    [kr, ka, kd, r0, theta0 / radian, phi0 / radian, alpha0 / radian, beta0 / radian, gamma0 / radian])
    return boresch


def _add_boresch_restraints(osys: System, gk: GlobalKeys):
    boresch_ligatoms = gk.boresch.ligatoms
    boresch_proatoms = gk.boresch.proatoms
    if boresch_ligatoms is None and boresch_proatoms is None:
        return
    k_kcal = gk.boresch.k_r_a_dih_kcal
    if gk.ab.ligatoms is not None:
        k_kcal[0] *= gk.ab.reslam
        k_kcal[1] *= gk.ab.reslam
        k_kcal[2] *= gk.ab.reslam
        logger.info(f"Lambda for restraints {gk.ab.reslam}")
    boresch_r_theta_phi = gk.boresch.r_theta_phi
    boresch_alpha_beta_gamma = gk.boresch.alpha_beta_gamma
    resforce = _make_boresch_restraints(boresch_ligatoms, boresch_proatoms, *k_kcal, *boresch_r_theta_phi,
                                        *boresch_alpha_beta_gamma)
    osys.addForce(resforce)
    logger.info(
        f"Boresch Restraints: lig3 {boresch_ligatoms} pro3 {boresch_proatoms} k_kcal {k_kcal} r,theta,phi {boresch_r_theta_phi} alpha,beta,gamma {boresch_alpha_beta_gamma}"
    )


########################################
# other forces -- Alchemical Nonbonded #
########################################


def _modify_torsion_force_by_vlambda(osys: System, gk: GlobalKeys):
    vlam = gk.ab.vlam
    if vlam == 1.0:
        return
    torsforce: PeriodicTorsionForce = get_force_from_system_by_name(osys, "PeriodicTorsionForce")
    if torsforce is None:
        return
    tors_changed = False
    ntors = torsforce.getNumTorsions()
    if gk.ab.ligatoms:
        n_previous_atoms = gk.ab.ligatoms[0]
        mol = Molecule(gk.filename.monomer)
        mg = MoleculeGraph(mol)
        linear_propers = mg.get_linear_propers()
        linear_set = set()
        for tf in linear_propers:
            p1, p2, p3, p4 = tf
            p2, p3 = p2 + n_previous_atoms, p3 + n_previous_atoms
            linear_set.add((p2, p3))
            linear_set.add((p3, p2))
        tfp = mg.get_tfd_propers()  # torsion finger print 0-based index
        # tfp = mg.get_rotatable_propers()
        p2p3_set = set()
        for tf in tfp:
            p1, p2, p3, p4 = tf
            p2, p3 = p2 + n_previous_atoms, p3 + n_previous_atoms
            p2p3_set.add((p2, p3))
            p2p3_set.add((p3, p2))
        for idx in range(ntors):
            p1, p2, p3, p4, nperiod, phase, k = torsforce.getTorsionParameters(idx)
            if ((p2, p3) in p2p3_set and (p2, p3) not in linear_set) or ((p3, p2) in p2p3_set and
                                                                         (p3, p2) not in linear_set):
                torsforce.setTorsionParameters(idx, p1, p2, p3, p4, nperiod, phase, k * vlam)
                tors_changed = True
    if not tors_changed:
        logger.info(f"Torsion vlam {vlam} not changed")


def _decouple_alchemical_group(natoms: int, nbforce: NonbondedForce, custom_nbforce: CustomNonbondedForce,
                               lj14: CustomBondForce, alchem_atoms, vlambda, elambda):
    alchem_particles = set(alchem_atoms)
    lig_pairs = dict()
    nexcept = nbforce.getNumExceptions()
    # exclude these pairs to 0 in nbforce
    for idx in range(nexcept):
        p1, p2, combinedCharge, combinedSigma, combinedEpsilon = nbforce.getExceptionParameters(idx)
        if p1 in alchem_atoms and p2 in alchem_atoms:
            lig_pairs[(p1, p2)] = (combinedSigma, combinedEpsilon)
            nbforce.setExceptionParameters(idx, p1, p2, combinedCharge * elambda, combinedSigma, combinedEpsilon * 0)
    # add 1-4+ interaction
    vlam2 = vlambda  # intra: annihilate
    # vlam2 = 1.0      # intra: decouple
    for p1 in alchem_particles:
        for p2 in alchem_particles:
            if p1 != p2:
                pkey = (p1, p2)
                if pkey in lig_pairs.keys():
                    sig, eps = lig_pairs[pkey]
                    if eps.value_in_unit(unit.kilojoule_per_mole) == 0.0:
                        pass  # 1-2, 1-3
                    else:
                        lj14.addBond(p1, p2, [sig, eps, vlam2, vlam2])  # 1-4
                elif p1 < p2:
                    # beyond 1-4
                    _charge1, sigma1, epsilon1 = nbforce.getParticleParameters(p1)
                    _charge2, sigma2, epsilon2 = nbforce.getParticleParameters(p2)
                    sig = (sigma1 + sigma2) * 0.5
                    eps = (epsilon1 * epsilon2)**0.5
                    lj14.addBond(p1, p2, [sig, eps, vlam2, vlam2])
    for idx in alchem_particles:
        charge, sigma, epsilon = nbforce.getParticleParameters(idx)
        nbforce.setParticleParameters(idx, charge * elambda, sigma,
                                      epsilon * 0)  # scale charge by elambda, and set LJ to 0
    chemical_particles = set(range(natoms)) - alchem_particles
    custom_nbforce.addInteractionGroup(alchem_particles, chemical_particles)


def _make_alchemical_nonbonded_force(nbforce: NonbondedForce, natoms: int, alchem_atoms, vlambda, elambda, OHH_list):
    alchem_total_qold, alchem_total_qnew, alchem_total_qeps = 0.0, 0.0, 1.e-6
    if elambda < 1.0:
        logger.info("modify NonbondedForce Charges")
        for idx in alchem_atoms:
            charge, sigma, epsilon = nbforce.getParticleParameters(idx)
            nbforce.setParticleParameters(idx, charge * elambda, sigma, epsilon)
            alchem_total_qold += charge.value_in_unit(unit.elementary_charge)
            alchem_total_qnew += charge.value_in_unit(unit.elementary_charge) * elambda
        alchem_qneeded = alchem_total_qold - alchem_total_qnew
        if abs(alchem_qneeded) > alchem_total_qeps:
            _alchemqneeded_cap = alchem_total_qold - 0.0
            nalchemwater = get_num_alchemwat(_alchemqneeded_cap)
            nwater = len(OHH_list)
            if nalchemwater > nwater:
                raise RuntimeError(f"AlchemWater: need at least {nalchemwater} water molecules; found {nwater}")
            alchem_q_mod = alchem_qneeded / nalchemwater
            if alchem_q_mod < 0:
                alchem_q_mod = alchem_q_mod / 2
            selected_OHH = OHH_list[-nalchemwater:]
            for iwat in selected_OHH:
                atom_O, atomH1, atomH2 = iwat
                q_O, sig_O, eps_O = nbforce.getParticleParameters(atom_O)
                qH1, sigH1, epsH1 = nbforce.getParticleParameters(atomH1)
                qH2, sigH2, epsH2 = nbforce.getParticleParameters(atomH2)
                q_O = q_O.value_in_unit(unit.elementary_charge)
                qH1 = qH1.value_in_unit(unit.elementary_charge)
                qH2 = qH2.value_in_unit(unit.elementary_charge)
                if alchem_q_mod > 0:
                    # make O less negative
                    nbforce.setParticleParameters(atom_O, q_O + alchem_q_mod, sig_O, eps_O)
                    logger.info(
                        f"AlchemWater atom and new charge: {atom_O} {atomH1} {atomH2} {q_O + alchem_q_mod} {qH1} {qH2}")
                elif alchem_q_mod < 0:
                    # make H less positive
                    nbforce.setParticleParameters(atomH1, qH1 + alchem_q_mod, sigH1, epsH1)
                    nbforce.setParticleParameters(atomH2, qH2 + alchem_q_mod, sigH2, epsH2)
                    logger.info(
                        f"AlchemWater atom and new charge: {atom_O} {atomH1} {atomH2} {q_O} {qH1 + alchem_q_mod} {qH2 + alchem_q_mod}"
                    )
        else:
            logger.info("AlchemWater: not needed")

        fudge_qq = 5. / 6.
        nexcept = nbforce.getNumExceptions()
        logger.info(f"Updating combined charge product 1-4 fudge {fudge_qq}")
        for idx in range(nexcept):
            p1, p2, combinedCharge, combinedSigma, combinedEpsilon = nbforce.getExceptionParameters(idx)
            charge1, _sigma1, _epsilon1 = nbforce.getParticleParameters(p1)
            charge2, _sigma2, _epsilon2 = nbforce.getParticleParameters(p2)
            if abs(combinedCharge.value_in_unit(unit.elementary_charge * unit.elementary_charge)) > 0.:
                nbforce.setExceptionParameters(idx, p1, p2, charge1 * charge2 * fudge_qq, combinedSigma,
                                               combinedEpsilon)

    if vlambda == 1.0:
        return None, None
    else:
        assert elambda == 0.0

    elj_cutoff = "select(step(r-ljcutoff), 0.0, eljpair); eljpair="
    elj_softcore = "vlam*4*epsilon*x*(x-1.0); x=sigmaPower6/reff_sterics6;"
    elj_reff_def = "reff_sterics6 = 0.5*(1.0-vlam)*sigmaPower6 + r^6; sigmaPower6 = sigma^6;"
    elj_combine_sigeps = "sigma=0.5*(sigma1+sigma2); epsilon=sqrt(epsilon1*epsilon2);"
    elj_combine_vlam = "vlam=min(vlam1,vlam2)"

    if nbforce.getNonbondedMethod() == NonbondedForce.NoCutoff:
        lj14 = CustomBondForce(elj_softcore + elj_reff_def + elj_combine_vlam)
    else:
        lj14 = CustomBondForce(elj_cutoff + elj_softcore + elj_reff_def + elj_combine_vlam)
        lj14.addGlobalParameter("ljcutoff", nbforce.getCutoffDistance())
    lj14.setUsesPeriodicBoundaryConditions(nbforce.usesPeriodicBoundaryConditions())
    lj14.addPerBondParameter("sigma")
    lj14.addPerBondParameter("epsilon")
    lj14.addPerBondParameter("vlam1")
    lj14.addPerBondParameter("vlam2")

    custom_nbforce = CustomNonbondedForce(elj_softcore + elj_reff_def + elj_combine_sigeps + elj_combine_vlam)
    custom_nbforce.addPerParticleParameter("sigma")
    custom_nbforce.addPerParticleParameter("epsilon")
    custom_nbforce.addPerParticleParameter("vlam")
    for idx in range(natoms):
        charge, sigma, epsilon = nbforce.getParticleParameters(idx)
        vlam = vlambda if idx in alchem_atoms else 1.0
        custom_nbforce.addParticle([sigma, epsilon, vlam])  # add LJ
    nexcept = nbforce.getNumExceptions()
    for idx in range(nexcept):
        p1, p2, _combinedCharge, _combinedSigma, _combinedEpsilon = nbforce.getExceptionParameters(idx)
        custom_nbforce.addExclusion(p1, p2)

    _decouple_alchemical_group(natoms, nbforce, custom_nbforce, lj14, alchem_atoms, vlambda, elambda)

    custom_nbforce.setUseSwitchingFunction(nbforce.getUseSwitchingFunction())
    custom_nbforce.setUseLongRangeCorrection(nbforce.getUseDispersionCorrection())
    if nbforce.getNonbondedMethod() == NonbondedForce.NoCutoff:
        custom_nbforce.setNonbondedMethod(CustomNonbondedForce.NoCutoff)
    elif nbforce.getNonbondedMethod() == NonbondedForce.CutoffNonPeriodic:
        custom_nbforce.setNonbondedMethod(CustomNonbondedForce.CutoffNonPeriodic)
        custom_nbforce.setCutoffDistance(nbforce.getCutoffDistance())
    else:
        custom_nbforce.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
        custom_nbforce.setCutoffDistance(nbforce.getCutoffDistance())
    return custom_nbforce, lj14


def _add_alchemical_nonbonded_force(osys: System, gk: GlobalKeys):
    ab_ligatoms = gk.ab.ligatoms
    if ab_ligatoms is None:
        return
    vlam = gk.ab.vlam
    elam = gk.ab.elam
    nbforce = get_force_from_system_by_name(osys, "NonbondedForce")
    natoms = osys.getNumParticles()
    _modify_torsion_force_by_vlambda(osys, gk)
    ab_nbforce, lig_lj14 = _make_alchemical_nonbonded_force(nbforce, natoms, ab_ligatoms, vlam, elam,
                                                            gk.filename.getListOfOHH())
    if ab_nbforce is not None:
        logger.info("add CustomNonbondedForce and CustomBondForce")
        osys.addForce(ab_nbforce)
        osys.addForce(lig_lj14)
    logger.info(f"ab_vlam {vlam} ab_elam {elam} ab_ligatoms {ab_ligatoms}")


########################################
# other forces -- Monte Carlo Barostat #
########################################


def _add_monte_carlo_barostat(osys: System, gk: GlobalKeys) -> None:
    itg = gk.integrator
    if not itg.npt:
        return
    targetP_bar = itg.targetP_bar
    targetT_K = itg.targetT_K
    freq = itg.npt_mc_freq
    baro = MonteCarloBarostat(targetP_bar, targetT_K, freq)
    osys.addForce(baro)
    logger.info(f"MonteCarloBarostat: {targetP_bar} bar {targetT_K} K per {freq} step(s)")


##########
# system #
##########


def get_force_from_system_by_name(osys: System, name: str) -> Optional[Force]:
    for f in osys.getForces():
        if f.getName() == name:
            return f
    return None


def _get_system_topology(gk: GlobalKeys) -> tuple[System, Topology]:
    if gk.openmm.params_ecosystem == "gromacs":
        if gk.filename.crd.endswith(".pdb"):
            pdbfile = _get_pdb(gk.filename.crd)
            pbcvecs = pdbfile.topology.getPeriodicBoxVectors()
        elif gk.filename.crd.endswith(".gro"):
            grofile = _get_gro(gk.filename.crd)
            pbcvecs = grofile.getPeriodicBoxVectors()
        else:
            assert False
        assert gk.filename.sys.endswith(".top") and Path(gk.filename.sys).exists()
        topfile = GromacsTopFile(gk.filename.sys, periodicBoxVectors=pbcvecs)
        osys: System = topfile.createSystem(nonbondedMethod=ff.PME,
                                            nonbondedCutoff=1.0 * unit.nanometer,
                                            constraints=ff.HBonds,
                                            rigidWater=True,
                                            ewaldErrorTolerance=0.0005,
                                            removeCMMotion=False,
                                            hydrogenMass=None,
                                            switchDistance=None)
        nbforce: NonbondedForce = get_force_from_system_by_name(osys, "NonbondedForce")
        if nbforce:
            nbforce.setUseDispersionCorrection(False)
        return osys, topfile.topology
    else:
        raise NotImplementedError(f"unknown params_ecosystem {gk.openmm.params_ecosystem}")


def get_system_topology(gk: GlobalKeys) -> tuple[System, Topology]:
    osys, otop = _get_system_topology(gk)
    _add_position_restraints(osys, gk)
    _add_boresch_restraints(osys, gk)

    _add_alchemical_nonbonded_force(osys, gk)

    _set_unique_forcegroup(osys)
    _add_monte_carlo_barostat(osys, gk)
    osys.addForce(CMMotionRemover())
    return osys, otop


def get_integrator(gk: GlobalKeys) -> Integrator:
    itg = gk.integrator
    name = itg.name
    dt_ps = itg.dt_ps
    randomseed = itg.randomseed
    if IntegratorNameOption(name) == IntegratorNameOption.LangevinMiddleIntegrator:
        targetT_K = itg.targetT_K
        friction_1_ps = itg.friction_1_ps
        oitg: Integrator = LangevinMiddleIntegrator(targetT_K, friction_1_ps, dt_ps)
        oitg.setConstraintTolerance(itg.constraint_tol)
        oitg.setRandomNumberSeed(randomseed)
        return oitg
    elif IntegratorNameOption(name) == IntegratorNameOption.BrownianIntegrator:
        targetT_K = 10.0
        friction_1_ps = 1000.
        dt_ps = 0.0001
        oitg: Integrator = BrownianIntegrator(targetT_K, friction_1_ps, dt_ps)
        return oitg
    elif IntegratorNameOption(name) == IntegratorNameOption.FIRE2:
        dt_start = 0.01 * unit.femtosecond
        dt_max = 0.5 * unit.femtosecond
        oitg: Integrator = FIRE2Integrator(dt_start, dt_max)
        return oitg
    raise NotImplementedError(name)


def get_simulation(gk: GlobalKeys) -> Simulation:
    osys, otop = get_system_topology(gk)
    oitg = get_integrator(gk)
    plt_str = gk.openmm.platform
    oplf: Platform = Platform.getPlatformByName(plt_str)
    cuda_prec = gk.openmm.precision
    osim: Simulation = Simulation(otop, osys, oitg, oplf, {"Precision": cuda_prec} if plt_str == "CUDA" else dict())
    initpos, initpbc = get_initial_pos_pbc(gk)
    osim.context.setPeriodicBoxVectors(*initpbc)
    osim.context.setPositions(initpos)
    targetT_K = gk.integrator.targetT_K
    integrator_randomseed = gk.integrator.randomseed
    if MinimizeRelaxOption(gk.integrator.minimize) == MinimizeRelaxOption.none:
        osim.context.setVelocitiesToTemperature(targetT_K, integrator_randomseed)
    return osim
