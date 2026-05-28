# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Tests for the felis-prolif bridge module and the prolif API contract.

Two categories of tests:
1. Interface contract tests -- verify the real prolif package still provides the
   API that felis expects, catching renamed methods, changed signatures, removed
   attributes, changed interaction names, or altered data structures before they
   silently break felis at runtime.
2. Bridge logic tests -- verify felis.utils.prolif_tools behaves correctly using
   the real prolif implementation.
"""

import inspect
import json
from pathlib import Path

import matplotlib
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import felis.external.prolif_tools as pt
from prolif import Fingerprint
from prolif import Molecule
from prolif.plotting.network import LigNetwork
from prolif.residue import ResidueId

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyFingerprintForKeys:

    def __init__(self, ifp):
        self.ifp = ifp


def _make_interacting_molecules():
    """Create a minimal ligand (ethanol) and protein (acetic acid) placed
    within H-bond distance of each other. Returns (lig_mol, prot_mol)."""
    lig = Chem.MolFromSmiles("CCO")
    lig = Chem.AddHs(lig)
    AllChem.EmbedMolecule(lig, randomSeed=42)

    prot = Chem.MolFromSmiles("CC(=O)O")
    prot = Chem.AddHs(prot)
    AllChem.EmbedMolecule(prot, randomSeed=42)

    # Translate prot to be ~3 A from lig
    lig_conf = lig.GetConformer()
    prot_conf = prot.GetConformer()
    lig_center = np.mean([list(lig_conf.GetAtomPosition(i)) for i in range(lig.GetNumAtoms())], axis=0)
    prot_center = np.mean([list(prot_conf.GetAtomPosition(i)) for i in range(prot.GetNumAtoms())], axis=0)
    translation = lig_center - prot_center + np.array([3.0, 0.0, 0.0])
    for i in range(prot.GetNumAtoms()):
        pos = list(prot_conf.GetAtomPosition(i))
        prot_conf.SetAtomPosition(i, (pos[0] + translation[0], pos[1] + translation[1], pos[2] + translation[2]))

    return Molecule.from_rdkit(lig), Molecule.from_rdkit(prot)


def _make_hydrophobic_pair():
    """Create two short alkane fragments (propane each) placed close enough
    to register as hydrophobic contacts. Returns (lig_mol, prot_mol)."""
    lig = Chem.MolFromSmiles("CCC")
    lig = Chem.AddHs(lig)
    AllChem.EmbedMolecule(lig, randomSeed=7)

    prot = Chem.MolFromSmiles("CCC")
    prot = Chem.AddHs(prot)
    AllChem.EmbedMolecule(prot, randomSeed=11)

    lig_conf = lig.GetConformer()
    prot_conf = prot.GetConformer()
    lig_center = np.mean([list(lig_conf.GetAtomPosition(i)) for i in range(lig.GetNumAtoms())], axis=0)
    prot_center = np.mean([list(prot_conf.GetAtomPosition(i)) for i in range(prot.GetNumAtoms())], axis=0)
    translation = lig_center - prot_center + np.array([4.0, 0.0, 0.0])
    for i in range(prot.GetNumAtoms()):
        pos = list(prot_conf.GetAtomPosition(i))
        prot_conf.SetAtomPosition(i, (pos[0] + translation[0], pos[1] + translation[1], pos[2] + translation[2]))

    return Molecule.from_rdkit(lig), Molecule.from_rdkit(prot)


def _make_vdw_methane_pair(distance: float):
    """Create two methane molecules separated by the given center-to-center
    distance (Angstrom). Returns (lig_mol, prot_mol)."""
    lig = Chem.MolFromSmiles("C")
    lig = Chem.AddHs(lig)
    AllChem.EmbedMolecule(lig, randomSeed=1)

    prot = Chem.MolFromSmiles("C")
    prot = Chem.AddHs(prot)
    AllChem.EmbedMolecule(prot, randomSeed=2)

    lig_conf = lig.GetConformer()
    prot_conf = prot.GetConformer()
    lig_center = np.mean([list(lig_conf.GetAtomPosition(i)) for i in range(lig.GetNumAtoms())], axis=0)
    prot_center = np.mean([list(prot_conf.GetAtomPosition(i)) for i in range(prot.GetNumAtoms())], axis=0)
    translation = lig_center - prot_center + np.array([distance, 0.0, 0.0])
    for i in range(prot.GetNumAtoms()):
        pos = list(prot_conf.GetAtomPosition(i))
        prot_conf.SetAtomPosition(i, (pos[0] + translation[0], pos[1] + translation[1], pos[2] + translation[2]))

    return Molecule.from_rdkit(lig), Molecule.from_rdkit(prot)


# ===================================================================
# Interface contract tests (real prolif, no monkeypatching)
# ===================================================================


class TestProlifSignatures:

    def test_fingerprint_constructor_signature(self):
        """Fingerprint.__init__ accepts interactions, parameters, count, vicinity_cutoff."""
        sig = inspect.signature(Fingerprint.__init__)
        param_names = set(sig.parameters.keys())
        for name in ("interactions", "parameters", "count", "vicinity_cutoff"):
            assert name in param_names, f"Fingerprint.__init__ missing parameter: {name}"

        fp = Fingerprint(interactions=["HBDonor"], parameters=None, count=False, vicinity_cutoff=6.0)
        assert fp.count is False
        assert fp.vicinity_cutoff == 6.0

    def test_fingerprint_run_signature(self):
        """Fingerprint.run accepts traj, lig, prot, n_jobs parameters, all callable
        by keyword as felis does, and n_jobs has a default value."""
        sig = inspect.signature(Fingerprint.run)
        param_names = set(sig.parameters.keys())
        for name in ("traj", "lig", "prot", "n_jobs"):
            assert name in param_names, f"Fingerprint.run missing parameter: {name}"

        # felis always calls fp.run with these as keyword arguments, so they must
        # not become positional-only.
        keyword_compatible = {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
        for name in ("traj", "lig", "prot", "n_jobs"):
            kind = sig.parameters[name].kind
            assert kind in keyword_compatible, (
                f"Fingerprint.run parameter {name!r} is no longer keyword-callable (kind={kind})")

        # n_jobs must remain optional (have a default) -- felis sometimes relies on
        # passing an explicit int but the keyword must accept a default form too.
        assert sig.parameters["n_jobs"].default is not inspect.Parameter.empty, \
            "Fingerprint.run.n_jobs lost its default value"

    def test_fingerprint_run_accepts_tuple_lig(self):
        """Fingerprint.run internally handles lig as a 2-tuple (sdf_file, mda_ag).
        This is a vendored-only patch (not in upstream prolif 2.0.3 / 2.1.0); it
        must be preserved so felis's call ``fp.run(lig=(lig_sdf, mda_ligand))`` works.
        """
        source = inspect.getsource(Fingerprint.run)
        assert "isinstance(lig, tuple)" in source, ("Fingerprint.run no longer handles lig as a tuple -- "
                                                    "felis passes lig=(lig_sdf, mda_ligand) to fp.run()")
        # The tuple branch must dispatch through Molecule.from_lig_sdf_mda; without
        # that, the tuple is recognised but nothing useful happens.
        assert "from_lig_sdf_mda" in source, ("Fingerprint.run no longer routes the tuple lig through "
                                              "Molecule.from_lig_sdf_mda -- felis depends on this vendored bridge")

    def test_fingerprint_plot_barcode_signature(self):
        """Fingerprint.plot_barcode accepts a dpi keyword argument."""
        sig = inspect.signature(Fingerprint.plot_barcode)
        assert "dpi" in sig.parameters, "Fingerprint.plot_barcode missing parameter: dpi"

    def test_fingerprint_list_available_is_static(self):
        """Fingerprint.list_available() can be called on the class and returns strings."""
        result = Fingerprint.list_available()
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, str) for x in result)

    def test_molecule_from_lig_sdf_mda_signature(self):
        """Molecule.from_lig_sdf_mda is a classmethod with sdf_file and mda_ag params.

        This classmethod is a vendored-only addition to prolif (not in upstream
        prolif 2.0.3 or 2.1.0); failure here typically indicates the vendored
        override has been lost or the submodule was replaced with a stock release.
        """
        sig = inspect.signature(Molecule.from_lig_sdf_mda)
        param_names = list(sig.parameters.keys())
        for name in ("sdf_file", "mda_ag"):
            assert name in param_names, f"Molecule.from_lig_sdf_mda missing parameter: {name}"

        # felis calls this positionally as Molecule.from_lig_sdf_mda(lig_sdf, mda_ligand);
        # any new parameters introduced upstream (e.g. sanitize, cleanup_substructures)
        # must therefore have defaults so the 2-arg form keeps working.
        assert param_names[0] == "sdf_file", \
            f"Molecule.from_lig_sdf_mda first parameter is no longer 'sdf_file': {param_names}"
        assert param_names[1] == "mda_ag", \
            f"Molecule.from_lig_sdf_mda second parameter is no longer 'mda_ag': {param_names}"
        keyword_compatible = {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        }
        for name in ("sdf_file", "mda_ag"):
            assert sig.parameters[name].kind in keyword_compatible, (
                f"Molecule.from_lig_sdf_mda.{name} is not positional-callable")
        for name in param_names[2:]:
            assert sig.parameters[name].default is not inspect.Parameter.empty, (
                f"Molecule.from_lig_sdf_mda gained a required parameter {name!r} -- "
                "felis's 2-arg call form would break")

        # Verify it's a classmethod
        method = inspect.getattr_static(Molecule, "from_lig_sdf_mda")
        assert isinstance(method, classmethod), "from_lig_sdf_mda is not a classmethod"

    def test_lignetwork_from_fingerprint_and_save_signatures(self):
        """LigNetwork.from_fingerprint accepts fp, ligand_mol, threshold.
        LigNetwork.save accepts a path argument (named ``fp`` in prolif source
        but interpreted as a filename / file-like object)."""
        from_fp_sig = inspect.signature(LigNetwork.from_fingerprint)
        from_fp_params = set(from_fp_sig.parameters.keys())
        for name in ("fp", "ligand_mol", "threshold"):
            assert name in from_fp_params, f"LigNetwork.from_fingerprint missing parameter: {name}"

        save_sig = inspect.signature(LigNetwork.save)
        save_params = [p for p in save_sig.parameters if p != "self"]
        assert "fp" in save_sig.parameters, "LigNetwork.save missing parameter: fp"
        # felis calls ``lignet.save(path)`` positionally; the first non-self
        # parameter must remain positional-or-keyword.
        first_param = save_sig.parameters[save_params[0]]
        assert first_param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.POSITIONAL_ONLY,
        ), f"LigNetwork.save first parameter is no longer positional-callable (kind={first_param.kind})"

    def test_vdwcontact_accepts_tolerance_parameter(self):
        """Fingerprint can be constructed with a per-interaction ``tolerance``
        parameter for VdWContact.

        felis.utils.prolif_tools._find_vdw_interactions_with_tols iterates over
        tolerances of 0.5, 1.0, 1.5 angstrom. Upstream prolif 2.1.0 adds a new
        ``preset`` parameter for VdWContact; if that ever displaces ``tolerance``
        or makes it mandatory, this test fails before felis breaks silently.
        """
        fp = Fingerprint(interactions=["VdWContact"], parameters={"VdWContact": {"tolerance": 0.5}})
        assert fp is not None

    def test_vdwcontact_tolerance_affects_detection(self):
        """Increasing the VdWContact tolerance must produce at least as many
        contacts as a small tolerance. If upstream prolif 2.1.0 ever makes
        ``tolerance`` a no-op (e.g., because a new ``preset`` overrides it),
        felis's VdW tolerance fallback chain becomes meaningless."""
        lig_mol, prot_mol = _make_vdw_methane_pair(4.5)
        fp_low = Fingerprint(interactions=["VdWContact"], parameters={"VdWContact": {"tolerance": 0.0}})
        ifp_low = fp_low.generate(lig_mol, prot_mol, metadata=True)
        fp_high = Fingerprint(interactions=["VdWContact"], parameters={"VdWContact": {"tolerance": 1.5}})
        ifp_high = fp_high.generate(lig_mol, prot_mol, metadata=True)

        def _count_vdw(ifp):
            count = 0
            for interactions in ifp.values():
                for name, metadata_tuple in interactions.items():
                    if name == "VdWContact":
                        count += len(metadata_tuple)
            return count

        low = _count_vdw(ifp_low)
        high = _count_vdw(ifp_high)
        assert high >= low, ("VdWContact tolerance no longer increases the number of detections "
                             f"(tolerance=0.0 -> {low}, tolerance=1.5 -> {high})")
        assert high > 0, ("VdWContact with tolerance=1.5 finds no contacts between two methanes at 4.5 A -- "
                          "the tolerance parameter may have become a no-op")


class TestInteractionNames:

    #: The felis-required "core set" of interactions. Every name here is
    #: referenced by felis.utils.prolif_tools.ProteinLigandInteraction and must
    #: remain present in Fingerprint.list_available() across prolif upgrades.
    felis_core_interactions = frozenset({
        "HBDonor",
        "HBAcceptor",
        "Anionic",
        "Cationic",
        "XBAcceptor",
        "XBDonor",
        "CationPi",
        "PiCation",
        "VdWContact",
        "EdgeToFace",
        "FaceToFace",
        "MetalAcceptor",
        "MetalDonor",
    })

    def test_all_felis_allowed_interactions_are_valid(self):
        """Every name in ProteinLigandInteraction.allowed_interactions is recognized by prolif."""
        available = set(Fingerprint.list_available())
        allowed = set(pt.ProteinLigandInteraction.allowed_interactions)
        missing = allowed - available
        assert not missing, f"Prolif no longer supports these interaction types: {missing}"

    def test_all_felis_default_interactions_are_valid(self):
        """Every name in ProteinLigandInteraction.default_interactions is recognized by prolif.

        Also enforces that prolif 2.1.0's new ``WaterBridge`` interaction is not
        silently added to felis defaults: felis.analyze_gmx does not pass a water
        AtomGroup, so WaterBridge cannot be evaluated there.
        """
        available = set(Fingerprint.list_available())
        default = set(pt.ProteinLigandInteraction.default_interactions)
        missing = default - available
        assert not missing, f"Prolif no longer supports these interaction types: {missing}"
        assert "WaterBridge" not in default, ("WaterBridge was added to felis default_interactions but "
                                              "analyze_gmx does not supply a water selection")

    def test_prolif_list_available_contains_felis_core_set(self):
        """prolif.Fingerprint.list_available() must continue to expose the felis
        core interaction set. If upstream prolif 2.1.0 ever renames or removes
        any of these, felis's analyze_gmx call with these interactions will fail
        immediately at Fingerprint construction rather than silently producing
        an incomplete IFP."""
        available = set(Fingerprint.list_available())
        missing = self.felis_core_interactions - available
        assert not missing, ("prolif.Fingerprint.list_available() is missing interactions that "
                             f"felis relies on: {missing}")


class TestIFPDataStructure:

    def test_ifp_data_structure_format(self):
        """IFP returned by Fingerprint.generate has the structure felis expects:
        - keys are (ResidueId, ResidueId) tuples
        - values are dicts mapping interaction name -> tuple of metadata dicts
        - each metadata dict has "indices" with "ligand" and "protein" keys."""
        lig_mol, prot_mol = _make_interacting_molecules()
        fp = Fingerprint(interactions=["HBDonor", "HBAcceptor"])
        ifp = fp.generate(lig_mol, prot_mol, metadata=True)

        assert ifp is not None, "generate() returned None"
        assert len(ifp) > 0, "generate() returned empty IFP"

        seen_interaction_names = set()
        for residue_pair, interactions in ifp.items():
            assert isinstance(residue_pair, tuple), f"Expected tuple key, got {type(residue_pair)}"
            assert len(residue_pair) == 2, f"Expected 2-element tuple, got {len(residue_pair)}"
            assert isinstance(interactions, dict), f"Expected dict value, got {type(interactions)}"

            for int_name, metadata_tuple in interactions.items():
                assert isinstance(int_name, str), f"Expected str interaction name, got {type(int_name)}"
                assert isinstance(metadata_tuple, tuple), f"Expected tuple of metadata, got {type(metadata_tuple)}"
                seen_interaction_names.add(int_name)

                for metadata in metadata_tuple:
                    assert isinstance(metadata, dict), f"Expected dict metadata, got {type(metadata)}"
                    assert "indices" in metadata, f"Metadata missing 'indices' key: {metadata.keys()}"
                    indices = metadata["indices"]
                    assert "ligand" in indices, f"indices missing 'ligand' key: {indices.keys()}"
                    assert "protein" in indices, f"indices missing 'protein' key: {indices.keys()}"

        # Catch upstream HBond SMARTS rewrites (prolif 2.1.0) that would silently
        # zero-out felis IFPs on the canonical ethanol + acetic acid pair.
        assert seen_interaction_names & {"HBDonor", "HBAcceptor"}, (
            "Canonical ethanol + acetic acid pair no longer registers any HBDonor / HBAcceptor; "
            "upstream HBond SMARTS rewrite likely broke felis-relevant detection")

    def test_hbond_detection_on_canonical_pair(self):
        """The canonical ethanol + acetic acid pair must always register at least
        one HBDonor or HBAcceptor. This is a regression guard against the prolif
        2.1.0 HBond SMARTS rewrite removing felis-relevant detection."""
        lig_mol, prot_mol = _make_interacting_molecules()
        fp = Fingerprint(interactions=["HBDonor", "HBAcceptor"])
        ifp = fp.generate(lig_mol, prot_mol, metadata=True)
        names = {name for interactions in ifp.values() for name in interactions}
        assert names & {"HBDonor", "HBAcceptor"}, \
            f"Expected HBDonor or HBAcceptor on ethanol + acetic acid; got {names}"

    def test_hydrophobic_detection_on_canonical_pair(self):
        """Two propane fragments at ~4 A must register at least one Hydrophobic
        contact. This is a regression guard against the prolif 2.1.0 Hydrophobic
        SMARTS rewrite (which now excludes any carbon bonded to N/O/F)."""
        lig_mol, prot_mol = _make_hydrophobic_pair()
        fp = Fingerprint(interactions=["Hydrophobic"])
        ifp = fp.generate(lig_mol, prot_mol, metadata=True)
        names = {name for interactions in ifp.values() for name in interactions}
        assert "Hydrophobic" in names, \
            f"Expected Hydrophobic contact between two propanes at 4 A; got {names}"

    def test_ifp_residue_id_string_conversion(self):
        """ResidueId.__str__ returns a plain string (not a repr wrapper).
        Format must be compatible with _convert_dict_keys which does
        str(residue_pair[0]) + "-" + str(residue_pair[1])."""
        rid = ResidueId.from_string("ALA10")
        s = str(rid)
        assert s == "ALA10", f"Expected 'ALA10', got {s!r}"
        assert "ResidueId" not in s, f"ResidueId.__str__ looks like a repr: {s!r}"

        rid2 = ResidueId.from_string("LIG1")
        combined = str(rid) + "-" + str(rid2)
        assert combined == "ALA10-LIG1", f"Expected 'ALA10-LIG1', got {combined!r}"

        # Chained representation (with chain id) is what felis sees in
        # production for MDAnalysis-loaded systems. Keep its format stable.
        rid_chain = ResidueId.from_string("ALA10.A")
        assert str(rid_chain) == "ALA10.A", f"Expected 'ALA10.A', got {str(rid_chain)!r}"

        # Whitespace handling: prolif 2.0.3 does NOT strip whitespace and parses
        # "  ALA10  " as the unknown residue UNK0. prolif 2.1.0 strips whitespace
        # and parses it as ALA10. Pinning the 2.0.3 behavior here surfaces an
        # upgrade-induced ResidueId parsing change as a test failure rather than
        # a silent JSON-key drift in felis output.
        rid_ws = ResidueId.from_string("  ALA10  ")
        assert str(rid_ws) == "UNK0", \
            f"ResidueId whitespace parsing changed (was 'UNK0' in 2.0.3): {str(rid_ws)!r}"

    def test_residue_id_hashable_and_tuple_keyable(self):
        """ResidueId must remain hashable and usable as part of a tuple dict
        key. felis._convert_dict_keys iterates over fp.ifp where keys are
        (ResidueId, ResidueId) tuples; if upstream prolif 2.1.0 ever makes
        ResidueId unhashable or breaks tuple equality, felis would fail at
        runtime instead of producing valid JSON."""
        rid_a = ResidueId.from_string("ALA10")
        rid_l = ResidueId.from_string("LIG1")
        # Hashable
        hash(rid_a)
        hash(rid_l)
        # Usable as dict key
        d = {(rid_a, rid_l): 1}
        assert d[(rid_a, rid_l)] == 1
        # Equality is consistent across re-parsing
        assert ResidueId.from_string("ALA10") == rid_a


# Bridge logic tests (real prolif, no monkeypatching)
# ===================================================================


class TestProteinLigandInteraction:

    def test_init_default_and_single_interaction(self):
        pli = pt.ProteinLigandInteraction()
        assert pli.interactions == pt.ProteinLigandInteraction.default_interactions

        pli2 = pt.ProteinLigandInteraction("HBDonor")
        assert pli2.interactions == ["HBDonor"]

    def test_init_rejects_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid interactions"):
            pt.ProteinLigandInteraction(interactions=123)  # type: ignore[arg-type]

    def test_init_rejects_invalid_interaction_name(self):
        with pytest.raises(AssertionError, match="Invalid interactions"):
            pt.ProteinLigandInteraction(["NotARealInteraction"])

    def test_convert_dict_keys(self):
        fp = _DummyFingerprintForKeys(
            ifp={0: {
                ("ALA10", "LIG1"): {
                    "HBDonor": [{
                        "x": 1
                    }]
                },
                ("GLU11", "LIG1"): {
                    "Cationic": [{
                        "y": 2
                    }]
                },
            }})
        data = pt.ProteinLigandInteraction._convert_dict_keys(fp)  # pylint: disable=protected-access
        assert 0 in data
        assert "ALA10-LIG1" in data[0]
        assert "GLU11-LIG1" in data[0]

    def test_to_json_writes_file(self, tmp_path: Path):
        fp = _DummyFingerprintForKeys(ifp={0: {("A", "B"): {"HBDonor": [{"x": 1}]}}})
        out_path = tmp_path / "out.json"
        data = pt.ProteinLigandInteraction.to_json(fp, str(out_path))
        assert out_path.is_file()
        assert data[0]["A-B"]["HBDonor"][0]["x"] == 1
        # pretty json (indent=2) should contain newlines + two spaces
        txt = out_path.read_text(encoding="utf-8")
        assert "\n  \"0\"" in txt or "\n  0" in txt or "\n  \"A-B\"" in txt

    def test_to_json_does_not_write_when_suffix_is_not_json(self, tmp_path: Path):
        fp = _DummyFingerprintForKeys(ifp={0: {("A", "B"): {"HBDonor": [{"x": 1}]}}})
        out_path = tmp_path / "out.txt"
        data = pt.ProteinLigandInteraction.to_json(fp, str(out_path))
        assert data
        assert not out_path.exists()


class TestAnalyzeFramesByProlif:

    def test_returns_none_for_existing_empty_json(self, tmp_path: Path):
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "prolif.json"
        json_path.write_text("{}", encoding="utf-8")

        res = pt.analyze_frames_by_prolif(
            out_dir=str(out_dir),
            json_file="prolif.json",
            gmxtop="dummy.top",
            trj="dummy.xtc",
            lig_sdf="dummy.sdf",
            pro_select="protein",
            lig_select="resname LIG",
        )
        assert res is None

    def test_returns_none_for_existing_empty_file(self, tmp_path: Path):
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "prolif.json"
        json_path.write_text("", encoding="utf-8")

        res = pt.analyze_frames_by_prolif(
            out_dir=str(out_dir),
            json_file="prolif.json",
            gmxtop="dummy.top",
            trj="dummy.xtc",
            lig_sdf="dummy.sdf",
            pro_select="protein",
            lig_select="resname LIG",
        )
        assert res is None

    def test_returns_existing_nonempty_json(self, tmp_path: Path):
        out_dir = tmp_path / "out"
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "prolif.json"
        expected = {"0": {"RES1-RES2": {"HBDonor": [{"dummy": 1}]}}}
        json_path.write_text(json.dumps(expected), encoding="utf-8")

        res = pt.analyze_frames_by_prolif(
            out_dir=str(out_dir),
            json_file="prolif.json",
            gmxtop="dummy.top",
            trj="dummy.xtc",
            lig_sdf="dummy.sdf",
            pro_select="protein",
            lig_select="resname LIG",
        )
        assert res == expected
