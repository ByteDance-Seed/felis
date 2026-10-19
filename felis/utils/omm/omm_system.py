# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging
import math
import os
from pathlib import Path
import re
from typing import Optional

from openmm import BrownianIntegrator
from openmm import CMMotionRemover
from openmm import CustomTorsionForce
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
from openmm import Vec3
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


def _patch_openmm_gromacstopfile_hmr_water_resnames() -> None:
    """Patch OpenMM GromacsTopFile.createSystem in-memory for Gromacs water names.

    OpenMM's hydrogenMass handling skips rigid water by checking for residue
    name ``HOH``. FELIS Gromacs topologies can label water as ``SOL`` or
    ``CRW`` instead, so extend that check when HMR is enabled.
    """

    try:
        import openmm.app.gromacstopfile as gtf
    except Exception:
        return

    create_system = gtf.GromacsTopFile.createSystem
    if getattr(create_system, "_apply_hmr_water_resname_patch", False):
        return

    import inspect

    try:
        src = inspect.getsource(create_system)
    except Exception as exc:
        raise ImportError("Failed to inspect OpenMM GromacsTopFile.createSystem") from exc

    # ``inspect.getsource()`` returns a class method with the method itself
    # indented but its body indented one level further.  ``textwrap.dedent``
    # alone cannot remove that leading method indentation, so remove the
    # indentation of the ``def`` line explicitly before executing the method
    # at module scope.
    method_indent = len(src) - len(src.lstrip())
    canon = "".join(
        line[method_indent:] if line.strip() else line
        for line in src.replace("\r\n", "\n").splitlines(True)
    )
    lines = canon.splitlines(True)
    rigid_water_lines = [line for line in lines if "rigidWater" in line and "residue.name" in line and "HOH" in line]
    if rigid_water_lines and all("SOL" in line and "CRW" in line for line in rigid_water_lines):
        setattr(gtf.GromacsTopFile.createSystem, "_apply_hmr_water_resname_patch", True)
        return

    hoh_pattern = re.compile(r"residue\.name\s*==\s*(['\"])HOH\1")
    replacements = 0
    updated_lines = []
    for line in lines:
        if "rigidWater" in line and "residue.name" in line and "HOH" in line:
            line, nrep = hoh_pattern.subn("residue.name in ('HOH', 'SOL', 'CRW')", line)
            replacements += nrep
        updated_lines.append(line)

    if replacements == 0:
        srcfile = inspect.getsourcefile(create_system)
        srcfile_msg = f" from {srcfile}" if srcfile else ""
        raise ImportError(
            "Cannot apply HMR water resname patch: no rigidWater HOH residue-name checks "
            f"were found in OpenMM GromacsTopFile.createSystem{srcfile_msg}"
        )
    patched_lines = []
    for line in updated_lines:
        if line.lstrip().startswith("def createSystem("):
            patched_lines.append(line.replace("def createSystem", "def createSystem_hmr_water_resname_patch", 1))
        else:
            patched_lines.append(line)
    patched_src = "".join(patched_lines)

    exec(patched_src, gtf.__dict__)  # pylint: disable=exec-used
    patched_func = gtf.__dict__.get("createSystem_hmr_water_resname_patch")
    if patched_func is None or not callable(patched_func):
        raise ImportError("Failed to load patched createSystem_hmr_water_resname_patch from exec globals")
    gtf.GromacsTopFile.createSystem = patched_func
    setattr(patched_func, "_apply_hmr_water_resname_patch", True)
    logger.info("Patched OpenMM GromacsTopFile.createSystem for HMR water residue names; updated %d rigidWater HOH checks", replacements)


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


def _make_boresch_restraints(lig3, pro3, reslam, kr_kcal, kangle_kcal, kdih_kcal, r0, theta0, phi0, alpha0, beta0,
                             gamma0) -> CustomCompoundBondForce:
    radian = 180 / math.pi
    kr = kr_kcal * (unit.kilocalories_per_mole / unit.angstrom**2) / (unit.kilojoules_per_mole / unit.nanometer**2)
    ka = kangle_kcal * (unit.kilocalories_per_mole / unit.radian**2) / (unit.kilojoules_per_mole / unit.radian**2)
    kd = kdih_kcal * unit.kilocalories_per_mole / unit.kilojoules_per_mole

    e_translate = "0.5*kr * reslam *(r-r0)^2 + 0.5*ka * reslam *(theta-theta0)^2 + kd * reslam *(1-cos(phi-phi0)) + "
    e_rotate = "0.5*ka * reslam *(alpha-alpha0)^2 + kd * reslam *(1-cos(beta-beta0)) + kd * reslam *(1-cos(gamma-gamma0));"
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
    boresch.addGlobalParameter("reslam", reslam)
    logger.info(f"Lambda for restraints {reslam}")
    boresch.addBond([lig3[0], lig3[1], lig3[2], pro3[0], pro3[1], pro3[2]],
                    [kr, ka, kd, r0, theta0 / radian, phi0 / radian, alpha0 / radian, beta0 / radian, gamma0 / radian])
    return boresch


def _add_boresch_restraints(osys: System, gk: GlobalKeys):
    boresch_ligatoms = gk.boresch.ligatoms
    boresch_proatoms = gk.boresch.proatoms
    if boresch_ligatoms is None and boresch_proatoms is None:
        return
    k_kcal = gk.boresch.k_r_a_dih_kcal
    boresch_r_theta_phi = gk.boresch.r_theta_phi
    boresch_alpha_beta_gamma = gk.boresch.alpha_beta_gamma
    resforce = _make_boresch_restraints(boresch_ligatoms, boresch_proatoms, gk.ab.reslam, *k_kcal, *boresch_r_theta_phi,
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


def _add_alchem_torsion_force(osys: System, gk: GlobalKeys):
    torsforce: PeriodicTorsionForce = get_force_from_system_by_name(osys, "PeriodicTorsionForce")
    energy_expression = "torsionLambda * k * (1 + cos(n * theta - phase));"
    alchem_torsion_force = CustomTorsionForce(energy_expression)
    alchem_torsion_force.addGlobalParameter("torsionLambda", 1.0)
    alchem_torsion_force.addPerTorsionParameter("k")
    alchem_torsion_force.addPerTorsionParameter("n")
    alchem_torsion_force.addPerTorsionParameter("phase")

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
                # Disable the original torsion term by setting k to 0,
                # and add the same torsion to alchem_torsion_force with lambda scaling.
                torsforce.setTorsionParameters(idx, p1, p2, p3, p4, nperiod, phase, 0.0)
                alchem_torsion_force.addTorsion(p1, p2, p3, p4, [k, nperiod, phase])

                tors_changed = True
    if not tors_changed:
        logger.info(f"Torsion scale was not applied")
    else:
        logger.info(f"Torsion lambda schedule added with energy expression {energy_expression}")
        osys.addForce(alchem_torsion_force)

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

def _add_perturb_nonbonded_force(osys: System, gk: GlobalKeys, customLambdaSchedules: dict = None, softcore_alpha: float = 0.5):
    ab_ligatoms = gk.ab.ligatoms
    if ab_ligatoms is None:
        logger.info("No ligand atoms found, perturb_nonbonded_force not added.")
        return
    nbforce = get_force_from_system_by_name(osys, "NonbondedForce")
    natoms = osys.getNumParticles()
    _add_alchem_torsion_force(osys, gk)

    if customLambdaSchedules is not None:
        logger.info("use customLambdaSchedules to perturb nonbonded force.")
        logger.info(f"customLambdaSchedules: {customLambdaSchedules}")
        perturb_nb_force, coulomb_14_force, lj_14_force = _make_perturb_nonbonded_force(
                                                    nbforce,
                                                    natoms,
                                                    ab_ligatoms,
                                                    customLambdaSchedules,
                                                    gk.ab.ilam,
                                                    softcore_alpha,
                                                    gk.filename.getListOfOHH())
    elif len(gk.ab.lambda_schedules["lambdaA_charge"]) > 0:
        perturb_nb_force, coulomb_14_force, lj_14_force = _make_perturb_nonbonded_force(
                                                    nbforce,
                                                    natoms,
                                                    ab_ligatoms,
                                                    gk.ab.lambda_schedules,
                                                    gk.ab.ilam,
                                                    softcore_alpha,
                                                    gk.filename.getListOfOHH())
    # TODO: add support when lambda schedules are not available, for perturb nonbonded force.
    else:
        raise RuntimeError("No lambda schedule found in gk.ab.lambda_schedules.")

    if perturb_nb_force is not None:
        logger.info("add PerturbNonbondedForce and CustomBondForce for 1-4 interactions")
        osys.addForce(perturb_nb_force)
        osys.addForce(coulomb_14_force)
        osys.addForce(lj_14_force)

        for ii, force in enumerate(osys.getForces()):
            if isinstance(force, NonbondedForce):
                # Remove the original NonbondedForce since its interactions will be handled
                # by the PerturbNonbondedForce and the 1-4 CustomBondForces.
                osys.removeForce(ii)
                break
    else:
        raise RuntimeError("PerturbNonbondedForce is not added to the system.")


def _make_perturb_nonbonded_force(
    nbforce: NonbondedForce,
    natoms: int,
    alchem_atoms: list[int],
    lambda_schedules: dict,
    lambda_index: int,
    softcore_alpha: float = 1.0,
    OHH_list = None,
    enable_alchem_water: bool = True):
    from PerturbNonbondedPlugin import PerturbNonbondedForce
    alchem_total_qold, alchem_total_qnew, alchem_total_qeps = 0.0, 0.0, 1.e-6

    logger.info(f"Initialize perturb nonbonded force with lambda_index {lambda_index}")

    elamA_list = lambda_schedules["lambdaA_charge"]
    elamB_list = lambda_schedules["lambdaB_charge"]
    elamC_list = lambda_schedules["lambdaC_charge"]
    rest2_list = lambda_schedules["rest2_scale"]
    vlamA_list = lambda_schedules["lambdaA_vdw"]
    vlamB_list = lambda_schedules["lambdaB_vdw"]

    alchem_water_chargeB = dict() # key: atom index, value: charge for state B. Used for perturbing alchemical water charges.

    perturb_nonbonded_force = PerturbNonbondedForce()

    lig_total_charge = 0.0
    for idx in alchem_atoms:
        charge, sigma, epsilon = nbforce.getParticleParameters(idx)
        lig_total_charge += charge.value_in_unit(unit.elementary_charge)

    # At state B the ligand charge is zero. Shift selected waters by the
    # ligand charge so the static counterions from makebox remain compensated.
    alchem_q_needed = 0.0
    if enable_alchem_water and abs(lig_total_charge) > alchem_total_qeps:
        elamA_last = elamA_list[-1]
        alchem_q_needed = lig_total_charge * (1 - elamA_last)

    if abs(alchem_q_needed) > alchem_total_qeps:
        nalchemwater = get_num_alchemwat(alchem_q_needed)
        logger.info(f"AlchemWater: need {nalchemwater} water molecules for compensate {alchem_q_needed} charge")
        nwater = len(OHH_list)
        if nalchemwater > nwater:
            raise RuntimeError(f"AlchemWater: need at least {nalchemwater} water molecules; found {nwater}")
        alchem_q_mod = alchem_q_needed / nalchemwater
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
                alchem_water_chargeB[atom_O] = q_O + alchem_q_mod
                alchem_water_chargeB[atomH1] = qH1
                alchem_water_chargeB[atomH2] = qH2
                logger.info(f"AlchemWater atom {atom_O, atomH1, atomH2} chargeA: {q_O} {qH1} {qH2}")
                logger.info(f"AlchemWater atom {atom_O, atomH1, atomH2} chargeB: {q_O + alchem_q_mod} {qH1} {qH2}")
            elif alchem_q_mod < 0:
                # make H less positive
                alchem_water_chargeB[atom_O] = q_O
                alchem_water_chargeB[atomH1] = qH1 + alchem_q_mod
                alchem_water_chargeB[atomH2] = qH2 + alchem_q_mod
                logger.info(f"AlchemWater atom {atom_O, atomH1, atomH2} chargeA: {q_O} {qH1} {qH2}")
                logger.info(f"AlchemWater atom {atom_O, atomH1, atomH2} chargeB: {q_O} {qH1 + alchem_q_mod} {qH2 + alchem_q_mod}")


    for ii in range(natoms):
        charge, sigma, epsilon = nbforce.getParticleParameters(ii)
        if ii in alchem_atoms:
            chargeA = charge
            chargeB = 0
            sigmaA = sigma
            sigmaB = sigma
            epsilonA = epsilon
            epsilonB = 0
            isDummy = True
            isREST2 = False

            perturb_nonbonded_force.addParticle(chargeA, sigmaA, epsilonA, chargeB, sigmaB, epsilonB, isDummy, isREST2)
        elif enable_alchem_water and ii in alchem_water_chargeB.keys():
            logger.info(f"AlchemWater atom {ii} new chargeB {alchem_water_chargeB[ii]}")
            chargeA = charge
            chargeB = alchem_water_chargeB[ii] * unit.elementary_charge
            sigmaA = sigma
            sigmaB = sigma
            epsilonA = epsilon
            epsilonB = epsilon
            isDummy = False
            isREST2 = False

            perturb_nonbonded_force.addParticle(chargeA, sigmaA, epsilonA, chargeB, sigmaB, epsilonB, isDummy, isREST2)
        else:
            chargeA = charge
            chargeB = charge
            sigmaA = sigma
            sigmaB = sigma
            epsilonA = epsilon
            epsilonB = epsilon
            isDummy = False
            isREST2 = False

            perturb_nonbonded_force.addParticle(chargeA, sigmaA, epsilonA, chargeB, sigmaB, epsilonB, isDummy, isREST2)

    perturb_nonbonded_force.setSoftcoreAlpha(1.0)
    perturb_nonbonded_force.setNonbondedMethod(PerturbNonbondedForce.PME)
    perturb_nonbonded_force.setCutoffDistance(nbforce.getCutoffDistance())
    perturb_nonbonded_force.enableEnergyConstantCache(True)
    perturb_nonbonded_force.setLambdaSchedules(
        elamA_list,
        elamB_list,
        elamC_list,
        vlamA_list,
        vlamB_list,
        rest2_list
    )
    perturb_nonbonded_force.addGlobalParameter("replica_id", lambda_index)

    coulomb_14_force = CustomBondForce("step(cutoff-r)*Eq;Eq=ONE_4PI_EPS0*chargeprod/r;")
    coulomb_14_force.addPerBondParameter("chargeprod")
    coulomb_14_force.addGlobalParameter("ONE_4PI_EPS0", 138.93545764438198)  # in OpenMM unit system: kJ/mol*nm/e^2
    coulomb_14_force.addGlobalParameter("cutoff", nbforce.getCutoffDistance().value_in_unit(unit.nanometer))
    coulomb_14_force.setUsesPeriodicBoundaryConditions(nbforce.usesPeriodicBoundaryConditions())

    lj_14_force = CustomBondForce("step(cutoff-r)*Eq;Eq=4*epsilon*((sigma/r)^12-(sigma/r)^6);")
    lj_14_force.addPerBondParameter("sigma")
    lj_14_force.addPerBondParameter("epsilon")
    lj_14_force.addGlobalParameter("cutoff", nbforce.getCutoffDistance().value_in_unit(unit.nanometer))
    lj_14_force.setUsesPeriodicBoundaryConditions(nbforce.usesPeriodicBoundaryConditions())

    num_exceptions = nbforce.getNumExceptions()
    for idx in range(num_exceptions):
        p1, p2, combinedCharge, combinedSigma, combinedEpsilon = nbforce.getExceptionParameters(idx)
        # TODO: Consider the rest2 strategy for 1-4 interactions in the future.
        if combinedCharge != 0.0:
            coulomb_14_force.addBond(p1, p2, [combinedCharge])
        if combinedEpsilon != 0.0:
            lj_14_force.addBond(p1, p2, [combinedSigma, combinedEpsilon])
        # The perturb_nonbonded_force will not handle 1-4 interactions, so we set the parameters to make the energy contribution zero.
        # The actual 1-4 interactions will be handled by the coulomb_14_force and lj_14_force.
        perturb_nonbonded_force.addException(p1, p2, 0, combinedSigma, 0)

    return perturb_nonbonded_force, coulomb_14_force, lj_14_force


###############################################
# other forces -- RelocationMC water sampling #
###############################################


# A hydrogen atom is treated as any particle lighter than this mass (amu). This
# matches bytemol's heavy-atom convention (atom.mass > 1.01 is heavy; see
# bytemol.toolkit.system_builder.system_builder_tools.get_index_summary_file).
_RELOCATIONMC_HEAVY_ATOM_MIN_MASS_AMU = 1.5


def _get_relocationmc_ligand_heavy_atoms(osys: System, ligatoms: list[int]) -> list[int]:
    """Return ligand heavy-atom indices (mass above the hydrogen threshold)."""
    heavy = []
    for idx in ligatoms:
        mass = osys.getParticleMass(idx).value_in_unit(unit.amu)
        if mass > _RELOCATIONMC_HEAVY_ATOM_MIN_MASS_AMU:
            heavy.append(idx)
    if not heavy:
        # Fall back to all ligand atoms if no heavy atom was identified so the
        # sphere centre is never empty.
        logger.warning("RelocationMC: no heavy ligand atom found; using all ligand atoms for centre.")
        heavy = list(ligatoms)
    return heavy


def _compute_relocationmc_sphere_radius(pos_nm, box_lengths_nm, ligand_heavy_atoms,
                                        scale: float, offset_nm: float) -> float:
    """Derive the inner sphere radius from ligand heavy-atom pairwise distances.

    Radius = scale * max_pairwise_distance(minimum-image) + offset_nm.

    PBC / conformation semantics:
      * Positions are taken from the *initial* coordinate file used to build the
        System (``gk.filename.crd``; for sysB this is the energy-minimized
        ``sysB_em.pdb``), i.e. the same frame that seeds every replica. The
        sphere is therefore a fixed geometric descriptor of the (relaxed) bound
        pose and does not change during the simulation.
      * Pairwise distances use the **minimum-image convention** on the
        orthorhombic periodic box, so the result is independent of where the
        ligand sits in the box and of any periodic wrapping of coordinates.
      * ``scale=0.5`` converts the maximum heavy-atom separation (a "diameter"
        of the ligand) into a radius, and ``offset_nm`` adds a solvent shell.
    """
    if len(ligand_heavy_atoms) < 2:
        # Single (or empty) heavy atom: no pairwise extent; use the offset only.
        return float(offset_nm)
    max_dist = 0.0
    coords = [pos_nm[i] for i in ligand_heavy_atoms]
    for ia in range(len(coords)):
        for ib in range(ia + 1, len(coords)):
            d2 = 0.0
            for axis in range(3):
                delta = coords[ia][axis] - coords[ib][axis]
                # minimum-image on the orthorhombic box
                length = box_lengths_nm[axis]
                if length > 0:
                    delta -= round(delta / length) * length
                d2 += delta * delta
            if d2 > max_dist:
                max_dist = d2
    max_pair = math.sqrt(max_dist)
    radius = scale * max_pair + offset_nm
    return float(radius)


def _get_relocationmc_alchemical_water_atoms(nbforce, ligatoms: list[int], OHH_list: list) -> set[int]:
    """Reproduce the PerturbNonbondedForce alchemical-water atom selection.

    Mirrors ``_make_perturb_nonbonded_force``: when the ligand is charged, the
    last ``get_num_alchemwat(|q_ligand|)`` water triples are alchemical. Their
    atom indices are returned so they can be excluded from the labile set.

    ``nbforce`` here is the PerturbNonbondedForce, whose Python wrapper returns
    particle parameters as plain numbers in (e, nm, kJ/mol).
    """
    def _charge_e(value):
        return float(value.value_in_unit(unit.elementary_charge)) if hasattr(value, "value_in_unit") else float(value)

    qeps = 1.e-6
    lig_total_charge = 0.0
    for idx in ligatoms:
        # PerturbNonbondedForce.getParticleParameters returns
        # (chargeA, sigmaA, epsilonA, chargeB, sigmaB, epsilonB, isDummy, isRest2).
        chargeA, _, _, *_ = nbforce.getParticleParameters(idx)
        lig_total_charge += _charge_e(chargeA)
    alchem_atoms: set[int] = set()
    if abs(lig_total_charge) <= qeps:
        return alchem_atoms
    alchem_q_needed = lig_total_charge * (-1.0)
    nalchemwater = get_num_alchemwat(alchem_q_needed)
    nwater = len(OHH_list)
    if nalchemwater > nwater:
        raise RuntimeError(f"RelocationMC: need at least {nalchemwater} water molecules; found {nwater}")
    for iwat in OHH_list[-nalchemwater:]:
        for atom in iwat:
            alchem_atoms.add(int(atom))
    logger.info(f"RelocationMC: charged ligand (q={lig_total_charge:.3f}e); "
                f"{nalchemwater} alchemical waters ({len(alchem_atoms)} atoms) excluded from labile set")
    return alchem_atoms


def _add_relocation_mc_force(osys: System, gk: GlobalKeys) -> None:
    """Add a RelocationMC force for water enhanced sampling (protein leg, sysB).

    Registered labile molecules are all solvent water molecules. When the ligand
    is charged, the alchemical charge-compensating waters are excluded from the
    labile set (they are not bulk-like and their charge is perturbed). The inner
    sphere is centred on the ligand heavy atoms with a radius derived from the
    ligand heavy-atom extent plus an offset.
    """
    rmc = gk.relocation_mc
    if not rmc.enable:
        return

    ligatoms = gk.ab.ligatoms
    if ligatoms is None:
        logger.info("RelocationMC: no ligand atoms found; force not added.")
        return

    OHH_list = gk.filename.getListOfOHH()
    if not OHH_list:
        logger.info("RelocationMC: no water molecules found (empty OHH list); force not added.")
        return

    try:
        from RelocationMCPlugin import LabileMolSiteParameters
        from RelocationMCPlugin import RelocationMC
    except ImportError as exc:
        raise ImportError(
            "RelocationMC is enabled (relocation_mc.enable=True) but the RelocationMCPlugin "
            "Python package could not be imported. Install the CUDA-enabled PerturbNonbondedPlugin "
            "first, then install RelocationMCPlugin with its install.sh, and ensure "
            "LD_LIBRARY_PATH/OPENMM_PLUGIN_DIR are set."
        ) from exc

    # The PerturbNonbondedForce is required by the RelocationMC CUDA kernels and
    # must already be present in the System (it is added just before this call).
    from PerturbNonbondedPlugin import PerturbNonbondedForce
    perturb_force = None
    for force in osys.getForces():
        if PerturbNonbondedForce.isinstance(force):
            perturb_force = PerturbNonbondedForce.cast(force)
            break
    if perturb_force is None:
        raise RuntimeError("RelocationMC requires a PerturbNonbondedForce in the System, but none was found.")

    natoms_per_water = len(OHH_list[0])
    if not all(len(w) == natoms_per_water for w in OHH_list):
        raise RuntimeError("RelocationMC requires every labile water to have the same number of sites")

    # Exclude alchemical (charge-compensating) waters for a charged ligand.
    excluded_atoms: set[int] = set()
    if rmc.exclude_alchemical_water:
        excluded_atoms = _get_relocationmc_alchemical_water_atoms(perturb_force, ligatoms, OHH_list)

    labile_waters = [w for w in OHH_list if not any(int(a) in excluded_atoms for a in w)]
    if not labile_waters:
        raise RuntimeError("RelocationMC: no labile water molecules remain after exclusions.")

    # Centre atoms = ligand heavy atoms.
    ligand_heavy_atoms = _get_relocationmc_ligand_heavy_atoms(osys, ligatoms)

    # Sphere radius from the initial minimized coordinates (minimum-image).
    init_pos, init_box = get_initial_pos_pbc(gk)
    pos_nm = init_pos.value_in_unit(unit.nanometer)
    box_vecs = init_box.value_in_unit(unit.nanometer)
    box_lengths = [box_vecs[0][0], box_vecs[1][1], box_vecs[2][2]]
    if rmc.sphere_radius_nm is not None:
        sphere_radius = float(rmc.sphere_radius_nm)
        logger.info(f"RelocationMC: using explicit sphere radius {sphere_radius:.4f} nm")
    else:
        sphere_radius = _compute_relocationmc_sphere_radius(
            pos_nm, box_lengths, ligand_heavy_atoms,
            scale=rmc.sphere_radius_scale, offset_nm=rmc.sphere_radius_offset_nm)
        logger.info(
            f"RelocationMC: derived sphere radius {sphere_radius:.4f} nm from "
            f"{len(ligand_heavy_atoms)} ligand heavy atoms "
            f"(scale={rmc.sphere_radius_scale}, offset={rmc.sphere_radius_offset_nm} nm)")

    temperature_K = rmc.temperature_K if rmc.temperature_K is not None else gk.integrator.targetT_K

    relocation = RelocationMC(temperature_K, sphere_radius,
                              int(rmc.exchange_frequency), int(rmc.num_candidate_sites))
    relocation.setRandomSeed(int(rmc.random_seed))
    relocation.setLigandAtoms([int(i) for i in ligand_heavy_atoms])

    # FELIS uses the symmetric multiple-try Metropolis (MTM) protocol. Keep
    # this order: the protocol setter validates the MTM prerequisites exposed
    # by the plugin. num_candidate_sites is the proposal-pool size K and is
    # passed directly to the constructor without the legacy 60x expansion.
    relocation.setUseCavityBias(False)
    relocation.setOrientationsEnabled(True)
    relocation.setUseLegacyAsymmetricEnergies(False)
    relocation.setProtocolVersion(RelocationMC.MTM)

    # Homogeneous site parameters from the first labile water, read from the
    # PerturbNonbondedForce (charge in e, sigma in nm, epsilon in kJ/mol). The
    # Python wrapper returns plain numbers; coerce units defensively.
    def _as_float(value, target_unit):
        return float(value.value_in_unit(target_unit)) if hasattr(value, "value_in_unit") else float(value)

    first_water = labile_waters[0]
    ref_atom = int(first_water[0])
    site_params = []
    for atom in first_water:
        charge, sigma, epsilon, *_ = perturb_force.getParticleParameters(int(atom))
        site_params.append(LabileMolSiteParameters(
            _as_float(charge, unit.elementary_charge),
            _as_float(sigma, unit.nanometer),
            _as_float(epsilon, unit.kilojoule_per_mole)))
    relocation.setLabileMolSiteParameters(site_params)

    # Rigid reference positions relative to the water oxygen (reference atom),
    # minimum-image so the rigid template is whole regardless of wrapping.
    ref_pos = pos_nm[ref_atom]
    ref_positions = []
    for atom in first_water:
        offsets = []
        for axis in range(3):
            delta = pos_nm[int(atom)][axis] - ref_pos[axis]
            length = box_lengths[axis]
            if length > 0:
                delta -= round(delta / length) * length
            offsets.append(delta)
        ref_positions.append(Vec3(offsets[0], offsets[1], offsets[2]))
    relocation.setLabileMolReferencePositions(ref_positions)

    for water in labile_waters:
        atom_indices = [int(a) for a in water]
        relocation.addLabileMol(atom_indices, atom_indices[0])

    if rmc.verbose_log:
        # HREMD launches one process per rank; make the log filename unique so
        # ranks do not contend for the same file. NOTE: in hremd the system is
        # built in the main process and forked to workers, so a main-pid suffix
        # is NOT unique across ranks — pass a per-rank path via rmc.verbose_log.
        log_path = str(rmc.verbose_log)
        stem, suffix = os.path.splitext(log_path)
        rank_log = f"{stem}.pid{os.getpid()}{suffix or '.log'}"
        relocation.setVerbose(True)
        relocation.setLogFileName(rank_log)
        logger.info(f"RelocationMC: verbose relocation log per-rank: {rank_log}")

    osys.addForce(relocation)
    logger.info(
        f"RelocationMC: added MTM force with {len(labile_waters)} labile waters "
        f"(freq={rmc.exchange_frequency}, K={rmc.num_candidate_sites}, "
        f"radius={sphere_radius:.4f} nm, T={temperature_K} K)")


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
        hmr_mass = None
        if gk.integrator.hmr is not None:
            _patch_openmm_gromacstopfile_hmr_water_resnames()
            hmr_mass = gk.integrator.hmr * unit.amu
        topfile = GromacsTopFile(gk.filename.sys, periodicBoxVectors=pbcvecs)
        osys: System = topfile.createSystem(nonbondedMethod=ff.PME,
                                            nonbondedCutoff=1.0 * unit.nanometer,
                                            constraints=ff.HBonds,
                                            rigidWater=True,
                                            ewaldErrorTolerance=0.0005,
                                            removeCMMotion=False,
                                            hydrogenMass=hmr_mass,
                                            switchDistance=None)
        nbforce: NonbondedForce = get_force_from_system_by_name(osys, "NonbondedForce")
        if nbforce:
            nbforce.setUseDispersionCorrection(False)
        return osys, topfile.topology
    else:
        raise NotImplementedError(f"unknown params_ecosystem {gk.openmm.params_ecosystem}")


def get_system_topology(gk: GlobalKeys, enable_perturb_nonbonded: bool = False, customLambdaSchedules: dict = None) -> tuple[System, Topology]:
    osys, otop = _get_system_topology(gk)
    _add_position_restraints(osys, gk)
    _add_boresch_restraints(osys, gk)

    if enable_perturb_nonbonded:
        _add_perturb_nonbonded_force(osys, gk, customLambdaSchedules)
        # RelocationMC (water enhanced sampling) requires the PerturbNonbondedForce
        # and is only supported on the perturb-nonbonded (sysB / v2) path.
        _add_relocation_mc_force(osys, gk)
    else:
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
