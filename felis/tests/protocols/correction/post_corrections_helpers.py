# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import math
from pathlib import Path

import numpy as np
import pandas as pd

from felis.protocols.correction.post_corrections import PostCorrections

TESTDATA_ROOT = Path(__file__).resolve().parents[2] / "testdata"

SHARED_HA_CONFIGS = [{
    "pka_tautomer": {
        "pH": 7.0,
        "pKa_list": [["a1", "a3", 7.0], ["a2", "a3", 7.0 - math.log10(3.0)]]
    }
}, {
    "pka_tautomer": {
        "pH": 7.0,
        "pKa_list": [["a1", "a3", 7.0]],
        "population": {
            "a1:a2": [0.25, 0.75],
        },
    }
}, {
    "pka_tautomer": {
        "pH": 7.0,
        "pKa_list": [["a2", "a3", 7.0 - math.log10(3.0)]],
        "population": {
            "a1:a2": [0.25, 0.75],
        },
    }
}, {
    "pka_tautomer": {
        "pH": 7.0,
        "population": {
            "a1:a2": [0.25, 0.75],
        },
        "pKa_list": [["a1", "a3", 7.0], ["a2", "a3", 7.0 - math.log10(3.0)]]
    }
}]

ALL_TAUTOMERS_CONFIGS = [{
    "pka_tautomer": {
        "population": {
            "a1:a2:a3": [0.3, 0.5, 0.2],
        },
    }
}, {
    "pka_tautomer": {
        "population": {
            "a1:a2": [0.375, 0.625],
            "a1:a3": [0.6, 0.4],
        },
    }
}]


def post_corrections_testdata_root() -> Path:
    return TESTDATA_ROOT / "stats" / "post_corrections"


def read_autosaved_results(pc, tag: str = "final") -> pd.DataFrame:
    """Trigger ``pc._autosave(tag)`` and read the produced TSV/CSV result.

    Centralizes the ``sep=None, engine="python"`` read so callers do not need to
    know the exact delimiter used by ``PostCorrections`` autosave artifacts.
    """
    res = pc._autosave(tag)
    return pd.read_csv(res, sep=None, engine="python")


def assert_single_merged_ligand_result(df: pd.DataFrame,
                                       *,
                                       ligand: str,
                                       expected_dg: float,
                                       abs_tol: float = 1e-10) -> None:
    """Assert the first result row corresponds to ``ligand`` with ``expected_dg``."""
    assert df["ligand"].iloc[0] == ligand, (f"Expected merged ligand label '{ligand}', got '{df['ligand'].iloc[0]}'")
    merged_dg = float(df["avg/dG"].iloc[0])
    assert math.isclose(
        merged_dg, expected_dg, rel_tol=0.0,
        abs_tol=abs_tol), (f"Merged dG mismatch: got {merged_dg}, expected {expected_dg} (abs_tol={abs_tol})")


def resolve_workflow_output_path(output_dir: Path, out_path: str) -> Path:
    """Resolve a ``PostCorrections.run`` return value to an existing path under ``output_dir``.

    Some workflow helpers return an absolute path while others return a path
    rooted under ``output_dir``; this normalizes both forms via ``Path``
    operations rather than POSIX-specific string splitting.
    """
    output_dir = Path(output_dir)
    candidate = Path(out_path)
    if candidate.is_absolute():
        if candidate.exists():
            return candidate
    else:
        joined = output_dir / candidate
        if joined.exists():
            return joined
    basename_joined = output_dir / candidate.name
    if basename_joined.exists():
        return basename_joined
    raise AssertionError(f"Could not resolve workflow output path under {output_dir}: {out_path!r}")


def assert_post_correction_node_df(df: pd.DataFrame) -> None:
    """Validate the schema and basic content of a post-correction node-wise dataframe."""
    required_cols = {"ligand", "avg/dG"}
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing required columns {missing} in node df (got {df.columns.tolist()})"
    assert len(df) > 0, "Node-wise post-correction dataframe is empty"
    ligand_series = df["ligand"].astype(str)
    assert (ligand_series.str.len() > 0).all(), "Encountered empty ligand label in post-correction dataframe"
    dg_values = pd.to_numeric(df["avg/dG"], errors="coerce")
    assert dg_values.notna().all(), "Encountered non-numeric avg/dG in post-correction dataframe"
    assert np.isfinite(
        dg_values.to_numpy(dtype=float)).all(), "Encountered non-finite avg/dG in post-correction dataframe"


def assert_ab_post_correction_artifacts(output_dir: Path, base: str) -> None:
    """Assert each AB intermediate correction stage materialized its TSV under ``output_dir``."""
    output_dir = Path(output_dir)
    expected = [
        f"{base}_01_symm.tsv",
        f"{base}_02_conf.tsv",
        f"{base}_03_pka.tsv",
        f"{base}_04_solvent.tsv",
        f"{base}_final.tsv",
    ]
    missing = [name for name in expected if not (output_dir / name).exists()]
    assert not missing, f"Missing AB post-correction artifacts under {output_dir}: {missing}"


def synthetic_component_dg_map() -> dict[str, float]:
    return {"a1": -1.0, "a2": 0.0, "a3": 1.0}


def build_synthetic_nodewise_df(dg_map: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame({
        "ligand": list(dg_map.keys()),
        "avg/dG": [float(v) for v in dg_map.values()],
    })


def run_synthetic_ab_pka_taut_correct(tmp_path: Path,
                                      cfg: dict,
                                      pka_rt: float = 0.596,
                                      *,
                                      stem: str) -> PostCorrections:
    dg_map = synthetic_component_dg_map()
    df = build_synthetic_nodewise_df(dg_map)
    csv_path = tmp_path / f"{stem}.tsv"
    df.to_csv(csv_path, sep="\t", index=False)

    pc = PostCorrections(str(csv_path), cfg, output_dir=str(tmp_path), rt=0.596, pka_rt=pka_rt)
    pc.pka_taut_correct(debug=False)
    return pc


def expected_shared_ha_merged_dg(pka_rt: float) -> float:
    dgs = synthetic_component_dg_map()
    ddg_13 = dgs["a3"] - dgs["a1"]
    ddg_12 = dgs["a2"] - dgs["a1"]
    num_val = 1.0 + np.exp(-ddg_13 / pka_rt) + 3.0 * np.exp(-ddg_12 / pka_rt)
    denum_val = 1.0 + 1.0 + 3.0
    return float(dgs["a1"] - pka_rt * np.log(num_val / denum_val))


def expected_shared_a_minus_merged_dg(pka_rt: float) -> float:
    dgs = synthetic_component_dg_map()
    ddg_13 = dgs["a3"] - dgs["a1"]
    ddg_12 = dgs["a2"] - dgs["a1"]
    num_val = 1.0 + np.exp(-ddg_13 / pka_rt) + 1.0 / 3.0 * np.exp(-ddg_12 / pka_rt)
    denum_val = 1.0 + 1.0 + 1.0 / 3.0
    return float(dgs["a1"] - pka_rt * np.log(num_val / denum_val))


def expected_all_tautomers_merged_dg(pka_rt: float) -> float:
    dgs = synthetic_component_dg_map()
    ddg_13 = dgs["a3"] - dgs["a1"]
    ddg_12 = dgs["a2"] - dgs["a1"]
    num_val = 1.0 + 2.0 / 3.0 * np.exp(-ddg_13 / pka_rt) + 5.0 / 3.0 * np.exp(-ddg_12 / pka_rt)
    denum_val = 1.0 + 2.0 / 3.0 + 5.0 / 3.0
    return float(dgs["a1"] - pka_rt * np.log(num_val / denum_val))


class _DSU:

    def __init__(self):
        self.parent: dict[str, str] = {}
        self.size: dict[str, int] = {}

    def add(self, x: str):
        x = str(x)
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1

    def find(self, x: str) -> str:
        x = str(x)
        if x not in self.parent:
            self.add(x)

        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def build_treatment_groups(cfg: dict, ligands_in_reference: list[str]) -> dict[str, str]:
    """Map ligand -> treatment group id derived from rbfe_correction_config.json."""
    dsu = _DSU()
    for lig in ligands_in_reference:
        dsu.add(lig)

    for grp in cfg.get("rotamer", []) or []:
        if not grp:
            continue
        states = [str(x) for x in grp]
        for st in states[1:]:
            dsu.union(states[0], st)

    pka = cfg.get("pka_tautomer") or {}
    pop = pka.get("population") or {}
    if isinstance(pop, dict):
        for key in pop.keys():
            states = [s for s in str(key).split(":") if s]
            if len(states) < 2:
                continue
            for st in states[1:]:
                dsu.union(states[0], st)

    for entry in pka.get("pKa_list", []) or []:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        dsu.union(str(entry[0]), str(entry[1]))

    return {lig: dsu.find(lig) for lig in ligands_in_reference}


def pick_group_representatives(ref_pred: pd.Series, group_map: dict[str, str], ref_order: list[str], *,
                               equal_tol: float) -> list[str]:
    """Pick one ligand per treatment group following the representative rules."""
    group_to_ligs: dict[str, list[str]] = {}
    for lig in ref_order:
        gid = group_map[lig]
        group_to_ligs.setdefault(gid, []).append(lig)

    selected: list[str] = []
    for gid, ligs in group_to_ligs.items():
        if len(ligs) == 1:
            selected.append(ligs[0])
            continue

        vals = ref_pred.loc[ligs].astype(float).to_numpy()
        max_val = np.max(vals)
        min_val = np.min(vals)
        if round(max_val - min_val, 2) <= equal_tol:
            selected.append(ligs[np.argmin(vals)])
            continue

        details = {lig: float(ref_pred.loc[lig]) for lig in ligs}
        raise AssertionError(f"Reference Pred dG values differ within the same treatment group {gid}: {details}, "
                             f"max={max_val}, min={min_val}, max - min > {equal_tol}")

    return selected


def expand_inhouse_output_to_parts(out_df: pd.DataFrame) -> pd.Series:
    """Expand a merged post-correction output frame to the original ligand parts."""
    if not {"ligand", "avg/dG", "ligand_parts"}.issubset(out_df.columns):
        raise AssertionError(f"Unexpected output schema: {out_df.columns.tolist()}")

    out_map: dict[str, float] = {}
    for _, row in out_df.iterrows():
        dg = float(row["avg/dG"])
        parts_raw = row["ligand_parts"]

        if isinstance(parts_raw, str):
            # This helper intentionally does NOT support parsing Python-repr
            # container strings produced by pandas `to_csv` round-trips.
            if parts_raw.lstrip().startswith(("[", "(", "{")):
                raise AssertionError("Unexpected stringified ligand_parts (looks like a serialized container). "
                                     "Do not read merged outputs from TSV/CSV if you need ligand_parts; "
                                     "use the in-memory `get_results()` dataframe instead.")
            parts_val = parts_raw
        else:
            parts_val = parts_raw

        parts: list[str] = []
        if isinstance(parts_val, (list, tuple)):
            for item in parts_val:
                if isinstance(item, (list, tuple)):
                    parts.extend([str(x) for x in item])
                else:
                    parts.append(str(item))
        else:
            parts = [str(parts_val)]

        for part in parts:
            if part in out_map and abs(out_map[part] - dg) > 1e-8:
                raise AssertionError(f"Inconsistent expanded dG for '{part}': {out_map[part]} vs {dg}")
            out_map[part] = dg

    return pd.Series(out_map, dtype=float)


def assert_sum0(values: np.ndarray, *, tol: float, context: str):
    total = float(np.sum(values))
    if abs(total) >= tol:
        raise AssertionError(f"In-house values are not sum-to-zero for {context}: sum={total}")
