# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import hashlib
import json
import math
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from felis.protocols.correction.post_corrections import PostCorrections
from felis.protocols.correction.post_corrections import RBPostCorrections

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


def rbfe_post_correction_root() -> Path:
    return TESTDATA_ROOT / "rbfe_post_correction"


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


def assert_rbfe_post_correction_artifacts(output_dir: Path, base: str) -> None:
    """Assert each RBFE intermediate correction stage materialized its TSV under ``output_dir``."""
    output_dir = Path(output_dir)
    expected = [
        f"{base}_edges_symm.tsv",
        f"{base}_edges_symm_ccc.tsv",
        f"{base}_edges_conf_shift_ccc.tsv",
        f"{base}_edges_pka.tsv",
        f"{base}_edges_final_edges.tsv",
    ]
    missing = [name for name in expected if not (output_dir / name).exists()]
    assert not missing, f"Missing RBFE post-correction artifacts under {output_dir}: {missing}"


def copy_packaged_rbfe_system(tmp_path: Path, *, system_subpath: str, dest_name: str) -> Path:
    """Copy a packaged RBFE reference system directory into ``tmp_path``.

    ``system_subpath`` is the path under :func:`rbfe_post_correction_root`
    (e.g. ``"merck/syk"``); ``dest_name`` is the directory name created under
    ``tmp_path``.
    """
    src = rbfe_post_correction_root() / system_subpath
    dest = Path(tmp_path) / dest_name
    shutil.copytree(src, dest, dirs_exist_ok=True)
    return dest


def load_ligand_predictions(sys_dir: Path) -> pd.DataFrame:
    """Read ``ligand_predictions.csv`` under ``sys_dir`` into a dataframe."""
    return pd.read_csv(Path(sys_dir) / "ligand_predictions.csv")


def load_rbfe_correction_config(sys_dir: Path) -> dict:
    """Load ``rbfe_correction_config.json`` under ``sys_dir`` as a dict."""
    return json.loads((Path(sys_dir) / "rbfe_correction_config.json").read_text(encoding="utf-8"))


def write_corrected_ligand_predictions(sys_dir: Path, df: pd.DataFrame) -> Path:
    """Write ``df`` as ``ligand_predictions_corrected.csv`` under ``sys_dir`` and return the path."""
    out_path = Path(sys_dir) / "ligand_predictions_corrected.csv"
    df.to_csv(out_path, index=False)
    return out_path


def build_synthetic_rbfe_edges(dg_map: dict[str, float]) -> pd.DataFrame:
    rows = []
    ligands = list(dg_map.keys())
    for idx, lig1 in enumerate(ligands):
        for lig2 in ligands[idx + 1:]:
            rows.append({
                "Lig 1": lig1,
                "Lig 2": lig2,
                "CCC ddG (kcal/mol)": float(dg_map[lig2]) - float(dg_map[lig1]),
            })
    return pd.DataFrame(rows)


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


def run_synthetic_rbfe_pka_taut_correct(tmp_path: Path,
                                        cfg: dict,
                                        pka_rt: float = 0.596,
                                        *,
                                        stem: str) -> RBPostCorrections:
    dg_map = synthetic_component_dg_map()
    edges = build_synthetic_rbfe_edges(dg_map)
    edges_path = tmp_path / f"{stem}.csv"
    edges.to_csv(edges_path, index=False)

    pc = RBPostCorrections(str(edges_path), cfg, output_dir=str(tmp_path), rt=0.596, pka_rt=pka_rt)
    pc._build_graph_from_consistent_edges(pc._base_edges)
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


def iter_rbfe_post_correction_system_dirs() -> list[Path]:
    cfgs = sorted(rbfe_post_correction_root().rglob("rbfe_correction_config.json"))
    return [p.parent for p in cfgs]


def reference_pred(ref_df: pd.DataFrame) -> pd.Series:
    """Reference Pred. dG series indexed by ligand name (no gauge shift)."""
    if "Ligand name" in ref_df.columns:
        lig_col = "Ligand name"
    elif "Ligand" in ref_df.columns:
        lig_col = "Ligand"
    else:
        raise AssertionError(f"Unexpected ligand_predictions ligand column: {ref_df.columns.tolist()}")

    pred_col = "Pred. dG (kcal/mol)"
    if pred_col not in ref_df.columns:
        raise AssertionError(f"Missing '{pred_col}' in ligand_predictions.csv")

    return pd.Series(
        pd.to_numeric(ref_df[pred_col], errors="raise").to_numpy(dtype=float),
        index=ref_df[lig_col].astype(str),
        dtype=float,
    )


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
    """Expand RBPostCorrections output to the original ligand parts."""
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
                                     "Do not read RBPostCorrections outputs from TSV/CSV if you need ligand_parts; "
                                     "use the in-memory `RBPostCorrections.get_results()` dataframe instead.")
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


def safe_sys_stem_for_tmp(sys_dir: Path) -> str:
    root = rbfe_post_correction_root()
    try:
        sys_rel = sys_dir.relative_to(root).as_posix()
    except ValueError:
        sys_rel = sys_dir.as_posix()

    safe = sys_rel.replace("/", "__").replace("\\", "__").replace(":", "_")
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", safe).strip("_")
    if len(safe) > 120:
        safe = safe[:120]

    digest = hashlib.sha1(sys_rel.encode("utf-8")).hexdigest()[:10]
    return f"rbfe_post_corr__{safe}__{digest}"


def dump_ligand_series_csv(tmp_path: Path, *, stem: str, tag: str, series: pd.Series) -> Path:
    out_path = tmp_path / f"{stem}__{tag}.csv"
    pd.DataFrame({
        "ligand": series.index.astype(str),
        "dG_kcal_mol": series.to_numpy(dtype=float),
    }).to_csv(out_path, index=False, float_format="%.8f")
    return out_path


def dump_df_tsv(tmp_path: Path, *, stem: str, tag: str, df: pd.DataFrame) -> Path:
    out_path = tmp_path / f"{stem}__{tag}.tsv"
    df.to_csv(out_path, sep="\t", index=False, float_format="%.8f")
    return out_path


def try_ref_dg_for_final_node_label(label: str, *, ref_pred: pd.Series, group_map: dict[str, str],
                                    group_rep: dict[str, str]) -> tuple[float, str | None]:
    label = str(label)
    if label in ref_pred.index:
        gid = group_map.get(label)
        if gid is None:
            return (float("nan"), f"node '{label}' exists in ref_pred but not in group_map")
        rep = group_rep.get(gid)
        if rep is None:
            return (float("nan"), f"missing representative for group '{gid}' (node '{label}')")
        return (float(ref_pred.loc[rep]), None)

    parts = [p for p in label.split("+") if p]
    if not parts:
        return (float("nan"), f"empty/invalid node label in final_edges: {label!r}")

    missing: list[str] = []
    gids: list[str] = []
    for part in parts:
        gid = group_map.get(part)
        if gid is None:
            missing.append(part)
        else:
            gids.append(gid)

    if missing and not gids:
        return (float("nan"), f"node '{label}' all parts are missing from reference/group_map: {missing}")

    uniq = sorted(set(gids))
    if len(uniq) != 1:
        return (float("nan"), f"node '{label}' spans multiple treatment groups: {uniq} (parts={parts})")

    rep = group_rep.get(uniq[0])
    if rep is None:
        return (float("nan"), f"missing representative for group '{uniq[0]}' (node '{label}')")
    return (float(ref_pred.loc[rep]), None)


def build_pka_component_map(cfg: dict) -> dict[str, str]:
    """Map ligand state -> pKa/tautomer connected-component label."""
    pka = cfg.get("pka_tautomer") or {}
    dsu = _DSU()

    pop = pka.get("population") or {}
    if isinstance(pop, dict):
        for key in pop.keys():
            states = [s for s in str(key).split(":") if s]
            if not states:
                continue
            dsu.add(states[0])
            for st in states[1:]:
                dsu.union(states[0], st)

    for entry in pka.get("pKa_list", []) or []:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        dsu.union(str(entry[0]), str(entry[1]))

    if not dsu.parent:
        return {}

    root_to_members: dict[str, list[str]] = {}
    for lig in list(dsu.parent.keys()):
        root = dsu.find(lig)
        root_to_members.setdefault(root, []).append(lig)

    out: dict[str, str] = {}
    for members in root_to_members.values():
        label = ":".join(sorted(members))
        for lig in members:
            out[lig] = label
    return out


def build_selected_ligand_diag_df(*, selected: list[str], out_sel_series: pd.Series, ref_raw_series: pd.Series,
                                  group_map: dict[str, str], group_rep: dict[str, str], cfg: dict) -> pd.DataFrame:
    pka_component_map = build_pka_component_map(cfg)
    rows = []
    for lig in selected:
        gid = group_map.get(lig)
        rep = group_rep.get(gid) if gid is not None else None
        out_dg = float(out_sel_series.loc[lig])
        ref_dg = float(ref_raw_series.loc[lig])
        delta = out_dg - ref_dg
        rows.append({
            "ligand": lig,
            "out_dG": out_dg,
            "ref_dG": ref_dg,
            "delta": delta,
            "abs_delta": abs(delta),
            "group_id": gid,
            "selected_representative": rep,
            "is_pka_member": lig in pka_component_map,
            "pka_component": pka_component_map.get(lig, ""),
        })
    return pd.DataFrame(rows).sort_values(["abs_delta", "ligand"], ascending=[False, True]).reset_index(drop=True)


def assert_rbfe_post_corrections_against_reference_csv(tmp_path: Path,
                                                       sys_dir: Path,
                                                       *,
                                                       in_csv: Path | None = None,
                                                       ref_csv: Path | None = None,
                                                       rt: float = 0.596,
                                                       pka_rt: float = 0.592484,
                                                       rotamer_route: str | None = "edge_shift_ccc",
                                                       tolerence: float = 0.01,
                                                       stem_tag: str | None = None,
                                                       skip_sum0: bool = False):
    """Assert RBFE post-correction results match a reference CSV.

    Runs ``RBPostCorrections`` on the input edge predictions, expands the
    output to per-ligand values, and compares against a reference
    ``ligand_predictions.csv`` (gauge-shifted to zero mean).

    Args:
        tmp_path: Temporary directory for intermediate artifacts.
        sys_dir: Root directory of the RBFE reference system (must contain
            ``edge_predictions.csv``, ``rbfe_correction_config.json``, and
            ``ligand_predictions.csv`` unless overridden).
        in_csv: Override for the input edge-predictions CSV path.
        ref_csv: Override for the reference ligand-predictions CSV path.
        rt: RT value passed to ``RBPostCorrections``.
        pka_rt: pKa RT value passed to ``RBPostCorrections``.
        rotamer_route: Rotamer correction route (e.g. ``"edge_shift_ccc"``).
        tolerence: Absolute tolerance for the per-ligand dG comparison
            against the reference.
        stem_tag: Optional tag appended to diagnostic file stems.
        skip_sum0: If ``True``, skip the strict sum-to-zero assertion on the
            corrected ligand dG values. This is intended for systems where
            the correction pipeline intentionally produces non-zero-sum
            results. All other numerical checks -- including the per-ligand
            ``assert_allclose`` comparison against the reference -- remain
            enforced regardless of this flag.
    """
    input_csv = in_csv or (sys_dir / "edge_predictions.csv")
    cfg_path = sys_dir / "rbfe_correction_config.json"
    ref_csv = ref_csv or (sys_dir / "ligand_predictions.csv")

    assert input_csv.exists(), f"Missing input: {input_csv}"
    assert cfg_path.exists(), f"Missing config: {cfg_path}"
    assert ref_csv.exists(), f"Missing reference: {ref_csv}"

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    pc = RBPostCorrections(str(input_csv), cfg, output_dir=str(tmp_path), rt=rt, pka_rt=pka_rt)
    stem = safe_sys_stem_for_tmp(sys_dir)
    if stem_tag:
        stem = f"{stem}__{stem_tag}"

    # System-specific expected failures are registered by the calling tests
    # (e.g. via ``pytest.mark.xfail`` in ``test_post_corrections_reference_batch``)
    # so this helper raises plain assertions and propagates exceptions normally.
    _ = pc.run(debug=False, rotamer_route=rotamer_route)

    # Use in-memory results to preserve structured columns like `ligand_parts`.
    out_df = pc.get_results()
    out_parts = expand_inhouse_output_to_parts(out_df)

    final_edges_path = tmp_path / f"{input_csv.stem}_edges_final_edges.tsv"
    assert final_edges_path.exists(), f"Missing RBPostCorrections final edges artifact: {final_edges_path}"
    final_edges_df = pd.read_csv(final_edges_path, sep="\t")
    required_edge_cols = {"Lig 1", "Lig 2", "ddG (kcal/mol)", "sigma"}
    if not required_edge_cols.issubset(final_edges_df.columns):
        missing_cols = sorted(required_edge_cols - set(final_edges_df.columns))
        raise AssertionError(
            f"Unexpected schema in {final_edges_path}: missing {missing_cols}, cols={final_edges_df.columns.tolist()}")
    generated_edges_dump = dump_df_tsv(tmp_path, stem=stem, tag="edges_final_edges_generated", df=final_edges_df)

    ref_df = pd.read_csv(ref_csv)
    ref_pred_series = reference_pred(ref_df)
    ref_order = ref_pred_series.index.astype(str).tolist()

    group_map = build_treatment_groups(cfg, ref_order)
    selected = pick_group_representatives(ref_pred_series, group_map, ref_order, equal_tol=1e-2)

    ref_sel = ref_pred_series.loc[selected].astype(float)
    ref0 = ref_sel - float(ref_sel.mean())

    group_rep: dict[str, str] = {}
    for lig in selected:
        gid = group_map.get(lig)
        if gid is None:
            raise AssertionError(f"Selected ligand '{lig}' missing from group_map")
        group_rep[gid] = lig

    missing = sorted(set(ref0.index) - set(out_parts.index))
    if missing:
        assert False, f"Reference contains ligands not present in edge network for {sys_dir}: {missing[:20]}"

    out_sel_series = out_parts.loc[ref0.index].astype(float)
    ref_raw_series = ref0.astype(float)
    out_sel_csv_path = dump_ligand_series_csv(tmp_path, stem=stem, tag="out_sel", series=out_sel_series)
    ref_raw_csv_path = dump_ligand_series_csv(tmp_path, stem=stem, tag="ref_raw", series=ref_raw_series)
    selected_diag_df = build_selected_ligand_diag_df(
        selected=selected,
        out_sel_series=out_sel_series,
        ref_raw_series=ref_raw_series,
        group_map=group_map,
        group_rep=group_rep,
        cfg=cfg,
    )
    selected_diag_dump = dump_df_tsv(tmp_path, stem=stem, tag="selected_diag", df=selected_diag_df)

    lig1 = final_edges_df["Lig 1"].astype(str)
    lig2 = final_edges_df["Lig 2"].astype(str)
    node_labels = sorted(set(lig1.tolist()) | set(lig2.tolist()))
    ref_dg_map: dict[str, float] = {}
    for label in node_labels:
        val, _ = try_ref_dg_for_final_node_label(
            label,
            ref_pred=ref_pred_series,
            group_map=group_map,
            group_rep=group_rep,
        )
        ref_dg_map[label] = float(val)

    ref_dg_lig1 = lig1.map(ref_dg_map).astype(float)
    ref_dg_lig2 = lig2.map(ref_dg_map).astype(float)
    ddg_ref = (ref_dg_lig2.to_numpy(dtype=float) - ref_dg_lig1.to_numpy(dtype=float)).astype(float)

    ref_edges_df = pd.DataFrame({
        "Lig 1": lig1,
        "Lig 2": lig2,
        "ref_dG_Lig1 (kcal/mol)": ref_dg_lig1,
        "ref_dG_Lig2 (kcal/mol)": ref_dg_lig2,
        "ddG_ref (kcal/mol)": ddg_ref,
    })
    ref_edges_dump = dump_df_tsv(tmp_path, stem=stem, tag="edges_ref_from_ligand_predictions", df=ref_edges_df)

    if "CCC ddG (kcal/mol)" in final_edges_df.columns:
        ccc_ddg = pd.to_numeric(final_edges_df["CCC ddG (kcal/mol)"], errors="raise").to_numpy(dtype=float)
    else:
        ccc_ddg = np.full(len(final_edges_df), np.nan, dtype=float)
    compare_edges_df = pd.DataFrame({
        "Lig 1": lig1,
        "Lig 2": lig2,
        "ref_dG_Lig1 (kcal/mol)": ref_dg_lig1,
        "ref_dG_Lig2 (kcal/mol)": ref_dg_lig2,
        "ddG_ref (kcal/mol)": ddg_ref,
        "CCC ddG (kcal/mol)": ccc_ddg,
        "delta (kcal/mol)": (ccc_ddg - ddg_ref),
    })
    compare_edges_dump = dump_df_tsv(tmp_path, stem=stem, tag="edges_final_edges_vs_ref", df=compare_edges_df)

    out_sel = out_sel_series.to_numpy(dtype=float)
    ref_raw = ref_raw_series.to_numpy(dtype=float)
    if not skip_sum0:
        assert_sum0(out_sel, tol=1e-2, context=str(sys_dir))

    try:
        np.testing.assert_allclose(out_sel - ref_raw, 0.0, atol=tolerence)
    except AssertionError as exc:
        raise AssertionError(f"{exc}\nDiagnostics written to:\n"
                             f"  {out_sel_csv_path}\n"
                             f"  {ref_raw_csv_path}\n"
                             f"  {selected_diag_dump}\n"
                             f"  {generated_edges_dump}\n"
                             f"  {ref_edges_dump}\n"
                             f"  {compare_edges_dump}\n") from exc


def integrate_population_reference_dg(dgs: list[float], weights: list[float], pka_rt: float) -> float:
    argmin_idx = int(np.argmin(dgs))
    min_dg = float(dgs[argmin_idx])
    min_w = float(weights[argmin_idx])
    num_term = 0.0
    den_term = 0.0
    for weight, dg in zip(weights, dgs):
        ratio = float(weight) / float(min_w)
        num_term += ratio * math.exp(-(float(dg) - float(min_dg)) / pka_rt)
        den_term += ratio
    return float(min_dg - pka_rt * math.log(num_term / den_term))


def integrate_pka_reference_dg(dg_deprot: float, dg_prot: float, *, pka_rt: float, pka_val: float, ph: float) -> float:
    ratio = 10**(pka_val - ph)
    exp_term = 10**((dg_deprot - dg_prot) / (pka_rt * math.log(10.0)))
    return float(dg_deprot - pka_rt * math.log((1.0 + ratio * exp_term) / (1.0 + ratio)))
