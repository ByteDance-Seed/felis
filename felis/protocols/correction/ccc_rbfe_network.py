# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CccResult:
    node_df: pd.DataFrame
    edge_df: pd.DataFrame
    diagnostics: dict


def read_rbfe_edges(path: str, *, sep: str | None = None) -> pd.DataFrame:
    """Read an RBFE edge table.

    This is a thin wrapper around pandas I/O.

    Args:
        path: Input CSV/TSV path.
        sep: If None, infer from extension (.tsv/.tab -> '\t', else ',').
    """
    if sep is None:
        lower = path.lower()
        if lower.endswith(".tsv") or lower.endswith(".tab"):
            sep = "\t"
        else:
            sep = ","
    return pd.read_csv(path, sep=sep)


def run_ccc_from_edge_df(
    edges_df: pd.DataFrame,
    *,
    lig1_col: str = "Lig 1",
    lig2_col: str = "Lig 2",
    ddg_col: str = "Bennett ddG (kcal/mol)",
    sigma_col: str = "Bennett std. error (kcal/mol)",
    strict: bool = True,
    out_lig1_col: str = "Lig 1",
    out_lig2_col: str = "Lig 2",
    out_ddg_col: str = "CCC ddG (kcal/mol)",
    out_sigma_col: str = "CCC std. error (kcal/mol)",
    debug: bool = False,
) -> CccResult:
    """Apply cycle-closure correction (CCC) to an RBFE edge network.

    Model:
      y_e = x_j - x_i + eps, eps ~ N(0, sigma_e^2)
    where e is a directed edge i -> j.

    For each connected component, solve WLS with a sum-to-zero gauge via a KKT system.

    Notation (shapes in parentheses):
      - y (m): observed directed edge ddGs, one per edge e: i->j
      - x (n): node potentials (relative dG values), one per ligand/node
      - B (m x n): incidence/design matrix; for edge i->j, the row has -1 at i and +1 at j
                 so (B x)_e = x_j - x_i
      - sigma (m): per-edge standard errors
      - W (m x m): diagonal weights, W = diag(1/(2*sigma^2))

    Under the Gaussian noise model, maximum likelihood is equivalent to weighted least squares:
      minimize_x  (y - Bx)^T W (y - Bx)

    Gauge constraint:
      x is only identifiable up to an additive constant (x + c*1 gives the same Bx), so we fix
      the reference level by enforcing 1^T x = 0.

    KKT (Karush-Kuhn-Tucker) linear system (equality-constrained WLS via Lagrange multiplier):
      [L, 1] [x]     [b]
      [1^T,0] [lambda] = [0]
    where L = B^T W B and b = B^T W y.

    The corrected (cycle-consistent) edge ddGs are then computed as:
      y_ccc = B x
    which implies exact cycle closure for any directed cycle in the same graph (telescoping sum).

    Duplicate directed edges (same lig1, lig2) in the *input* are rejected.

    Per the CCC implementation spec, the solver additionally requires the
    (undirected) network to have exactly one connected component.

    Args:
        edges_df: Input edge DataFrame.
        lig1_col, lig2_col, ddg_col, sigma_col: Input column names.
        strict: If True, validate aggressively and fail fast.
        out_*_col: Output column names for the copied input ddG/sigma columns.
    """
    required_cols = [lig1_col, lig2_col, ddg_col, sigma_col]
    missing = [c for c in required_cols if c not in edges_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = edges_df[[lig1_col, lig2_col, ddg_col, sigma_col]].copy()
    df = df.rename(columns={
        lig1_col: "_lig1",
        lig2_col: "_lig2",
        ddg_col: "_ddg",
        sigma_col: "_sigma",
    })

    if strict:
        if df["_lig1"].isna().any() or df["_lig2"].isna().any():
            raise ValueError("Ligand columns contain NaN")
        if (df["_lig1"].astype(str) == df["_lig2"].astype(str)).any():
            bad = df[df["_lig1"].astype(str) == df["_lig2"].astype(str)].iloc[0]
            raise ValueError(f"Self-edge detected (ligand maps to itself): {bad['_lig1']}")
        if df.duplicated(["_lig1", "_lig2"]).any():
            dup = df[df.duplicated(["_lig1", "_lig2"], keep=False)].sort_values(["_lig1", "_lig2"])
            raise ValueError("Duplicate directed edges detected for the same (Ligand 1, Ligand 2). "
                             f"First duplicates:\n{dup.head(10).to_string(index=False)}")

    df["_ddg"] = pd.to_numeric(df["_ddg"], errors="coerce")
    df["_sigma"] = pd.to_numeric(df["_sigma"], errors="coerce")
    if strict:
        if not np.isfinite(df["_ddg"]).all():
            raise ValueError("ddG column contains NaN/inf/non-numeric values")
        if not np.isfinite(df["_sigma"]).all():
            raise ValueError("sigma/stderr column contains NaN/inf/non-numeric values")
        if (df["_sigma"] <= 0).any():
            bad = df[df["_sigma"] <= 0].iloc[0]
            raise ValueError(
                f"sigma/stderr must be positive; got {bad['_sigma']} for edge {bad['_lig1']}->{bad['_lig2']}")

    # Work in string space for stable joins
    df["_lig1"] = df["_lig1"].astype(str)
    df["_lig2"] = df["_lig2"].astype(str)

    # Build undirected connectivity and require exactly one connected component.
    g = nx.Graph()
    nodes = pd.concat([df["_lig1"], df["_lig2"]], ignore_index=True).unique().tolist()
    g.add_nodes_from(nodes)
    g.add_edges_from(list(zip(df["_lig1"].tolist(), df["_lig2"].tolist())))
    components = [sorted(list(c)) for c in nx.connected_components(g)]
    if not components:
        raise ValueError("No ligands/edges found")
    if len(components) != 1:
        raise ValueError(f"Expected exactly 1 connected component, got {len(components)}: {components}")

    diagnostics: dict = {
        "n_nodes": int(len(nodes)),
        "components": [{
            "component_id": 0,
            "n_nodes": int(len(nodes)),
            "n_edges": int(len(df)),
        }],
    }

    # Solve CCC on the input edge set.
    comp_nodes = sorted(nodes)
    n_nodes = len(comp_nodes)
    n_edges = len(df)
    node_index = {name: idx for idx, name in enumerate(comp_nodes)}

    row_idx = np.repeat(np.arange(n_edges, dtype=int), 2)
    col_idx = np.empty(2 * n_edges, dtype=int)
    data = np.empty(2 * n_edges, dtype=float)
    lig1_idx = df["_lig1"].map(node_index).to_numpy(dtype=int)
    lig2_idx = df["_lig2"].map(node_index).to_numpy(dtype=int)
    col_idx[0::2] = lig1_idx
    col_idx[1::2] = lig2_idx
    data[0::2] = -1.0
    data[1::2] = 1.0
    b_mat = sp.coo_matrix((data, (row_idx, col_idx)), shape=(n_edges, n_nodes)).tocsr()
    if debug:
        logger.info(f"b_mat shape: {b_mat.shape}")
        logger.info(f"b_mat: {b_mat}")

    y = df["_ddg"].to_numpy(dtype=float)
    sigma = df["_sigma"].to_numpy(dtype=float)
    w = 0.5 / (sigma * sigma)
    sqrtw = np.sqrt(w)

    b_w = b_mat.multiply(sqrtw[:, None])
    l_mat = (b_w.T @ b_w).toarray()
    b_vec = (b_w.T @ (sqrtw * y)).reshape(-1)

    ones = np.ones(n_nodes, dtype=float)
    kkt = np.block([[l_mat, ones[:, None]], [ones[None, :], np.zeros((1, 1), dtype=float)]])
    rhs = np.concatenate([b_vec, np.array([0.0])])

    try:
        sol = np.linalg.solve(kkt, rhs)
    except np.linalg.LinAlgError as exc:
        msg = (f"CCC solve failed (n_nodes={n_nodes}, n_edges={n_edges}). "
               "The network may be underdetermined or ill-conditioned.")
        if strict:
            raise ValueError(msg) from exc
        logger.warning(msg)
        edge_out = pd.DataFrame({
            out_lig1_col: df["_lig1"],
            out_lig2_col: df["_lig2"],
            out_ddg_col: df["_ddg"],
            out_sigma_col: df["_sigma"],
            "CCC ddG (kcal/mol)": df["_ddg"],
        })
        return CccResult(node_df=pd.DataFrame(), edge_df=edge_out, diagnostics=diagnostics)

    x = sol[:n_nodes]
    cov_x = np.linalg.inv(kkt)[:n_nodes, :n_nodes]
    y_ccc = x[lig2_idx] - x[lig1_idx]

    y_ccc_input = np.array(y_ccc, dtype=float)

    edge_out = pd.DataFrame({
        out_lig1_col: df["_lig1"],
        out_lig2_col: df["_lig2"],
        out_ddg_col: df["_ddg"],
        out_sigma_col: df["_sigma"],
        "CCC ddG (kcal/mol)": y_ccc_input,
    })

    node_out = pd.DataFrame({
        "Ligand": comp_nodes,
        "component_id": [0] * n_nodes,
        "dG (kcal/mol)": x,
        "dG std. error (kcal/mol)": np.sqrt(np.maximum(np.diag(cov_x), 0.0)),
    })

    residual = y - y_ccc
    chi2 = float(np.sum((residual / sigma)**2))
    dof = int(n_edges - (n_nodes - 1))
    diagnostics["components"][0].update({
        "chi2": chi2,
        "dof": dof,
        "reduced_chi2": float(chi2 / dof) if dof > 0 else float("nan"),
        "sum_x": float(np.sum(x)),
    })
    diagnostics["n_components"] = 1
    diagnostics["n_edges"] = int(len(edge_out))
    diagnostics["n_nodes"] = int(len(node_out))
    return CccResult(node_df=node_out, edge_df=edge_out, diagnostics=diagnostics)
