# Copyright (c) 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import math
import os
from typing import Dict, Iterable

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


class BasePostCorrections:
    g: nx.Graph | None = None
    pH: float | None = None

    # Constants
    # https://github.com/schrodinger/public_binding_free_energy_benchmark/blob/main/fep_benchmark_inputs/fep_plus_inputs/merck/hif2a_symbmcorr.py#L73C9-L73C14
    RT = 0.596
    LN10 = math.log(10)

    def __init__(self,
                 input_csv: str,
                 correction_config: Dict,
                 output_dir: str = None,
                 *,
                 rt: float = RT,
                 pka_rt: float | None = None):
        """Shared attribute initialization for all post-correction workflows."""
        self.all_dgs_csv = input_csv
        self.correction_config = correction_config
        self.output_dir = output_dir
        self.g = nx.Graph()
        self.RT = float(rt)
        self.pka_rt = float(pka_rt) if pka_rt is not None else self.RT
        self.pH = None
        self.basename = os.path.splitext(self.all_dgs_csv)[0]

    def symm_correct(self, debug: bool = False):
        """
        Applies symmetry correction.
        dG = dG_init - RT * ln(n)
        """
        if 'symmetry' in self.correction_config:
            for lig, n in self.correction_config['symmetry'].items():
                if self.g.has_node(lig):
                    dG_init = self.g.nodes[lig]['dG']
                    dG_corr = dG_init - self.RT * math.log(n)
                    self.g.nodes[lig]['dG'] = dG_corr
                else:
                    raise ValueError(
                        f"Symmetry config mentions '{lig}', but it is not found in the input file {self.all_dgs_csv}")

        if debug:
            self._autosave('01_symm')

    def conf_correct(self, debug: bool = False):
        """
        Applies rotamer (conformer) correction.
        dG_merged = dG_base - RT * ln(sum_j exp(-(dG_j - dG_base) / RT))
        where dG_base is arbitrarily chosen as the minimum dG among the rotamers.
        """
        if 'rotamer' in self.correction_config:
            for rotamer_list in self.correction_config['rotamer']:
                # Filter to nodes that actually exist in the graph
                nodes_to_merge = [n for n in rotamer_list if self.g.has_node(n)
                                 ]  # order determined by correction_config['rotamer']
                if not nodes_to_merge:
                    raise ValueError(f"Rotamer list {rotamer_list} contains no valid nodes in the graph.")
                assert len(
                    nodes_to_merge) > 1, f"Rotamer list {rotamer_list} contains only one node, which is not allowed."
                assert len(set(nodes_to_merge)) == len(
                    nodes_to_merge), f"Nodes in graph to merge list {nodes_to_merge} contains duplicate nodes."
                assert len(
                    set(rotamer_list)) == len(rotamer_list), f"Rotamer list {rotamer_list} contains duplicate nodes."
                assert len(nodes_to_merge) == len(
                    rotamer_list
                ), f"len(nodes_to_merge) != len(rotamer_list): {len(nodes_to_merge)} != {len(rotamer_list)}"

                # Update parts right before merge
                for n in nodes_to_merge:
                    self.g.nodes[n]['avg_dG_parts'] = self.g.nodes[n]['dG']

                # Calculate merged dG
                dgs = [self.g.nodes[n]['dG'] for n in nodes_to_merge]
                min_dg = min(dgs)
                sum_exp = sum(math.exp(-(dg - min_dg) / self.RT) for dg in dgs)
                merged_dg = min_dg - self.RT * math.log(sum_exp)

                # Create merged node name
                merged_name = "+".join(nodes_to_merge)

                # Propagate experimental reference if present on any node.
                exp_vals = [self.g.nodes[n].get("exp") for n in nodes_to_merge]
                valid_exps = [e for e in exp_vals if e is not None]
                exp_out = None
                if valid_exps:
                    exp_out = float(valid_exps[0])
                    for e in valid_exps[1:]:
                        if not math.isclose(float(e), exp_out, abs_tol=1e-4):
                            raise ValueError(
                                f"Nodes in rotamer group {nodes_to_merge} have differing experimental values: {valid_exps}"
                            )

                # Collect attributes
                merged_ligand_parts = []
                merged_avg_dg_parts = []
                for n in nodes_to_merge:
                    merged_ligand_parts.append(self.g.nodes[n]['ligand_parts'])
                    merged_avg_dg_parts.append(self.g.nodes[n]['avg_dG_parts'])

                # Add new node
                self.g.add_node(merged_name,
                                dG=merged_dg,
                                exp=exp_out,
                                ligand_parts=merged_ligand_parts,
                                avg_dG_parts=merged_avg_dg_parts)

                # Remove old nodes
                self.g.remove_nodes_from(nodes_to_merge)

        if debug:
            self._autosave('02_conf')

    @staticmethod
    def _flatten_ligand_parts(parts) -> Iterable[str]:
        if isinstance(parts, str):
            yield parts
            return
        if isinstance(parts, (list, tuple)):
            for x in parts:
                yield from BasePostCorrections._flatten_ligand_parts(x)
            return
        # Unknown type: treat as scalar string representation.
        yield str(parts)

    def _orig_ligand_to_current_node(self) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for node, attrs in self.g.nodes(data=True):
            for part in self._flatten_ligand_parts(attrs.get("ligand_parts")):
                if part in mapping and mapping[part] != node:
                    raise ValueError(f"Ligand part '{part}' maps to multiple nodes: {mapping[part]} vs {node}")
                mapping[part] = node
        return mapping

    @staticmethod
    def _add_log_ratio_constraint(
        adjacency: dict[str, list[tuple[str, float, str]]],
        directed_constraints: dict[tuple[str, str], tuple[float, str]],
        src: str,
        dst: str,
        delta_log10: float,
        source_desc: str,
        *,
        tol: float = 1e-8,
    ):
        """Register one reversible log10-ratio constraint inside a component.

        Parameters
        ----------
        adjacency
            Directed adjacency list used later by `_solve_log10_component_weights`.
            Each stored edge means:

                log10(weight(dst)) - log10(weight(src)) = delta_log10

        directed_constraints
            Exact `(src, dst) -> (delta, source_desc)` registry used for
            self-consistency checks. If the same directed pair is introduced more
            than once, we require the two deltas to agree within `tol`; otherwise
            the input configuration is internally contradictory.

        delta_log10
            Relative population ratio written in base-10 log space. In this file
            we create these constraints from two sources:

            1. `population` tautomer groups
               `weight(i) / weight(anchor) = w_i / w_anchor`
               so `delta_log10 = log10(w_i / w_anchor)`.
            2. `pKa_list` micro-pKa relations
               `weight(HA) / weight(A-) = 10^(pKa - pH)`
               so `delta_log10 = pKa - pH`.

        Why store the reverse edge automatically
        ---------------------------------------
        The solver walks the connected component in arbitrary direction. Adding
        both `src -> dst` and `dst -> src` lets the traversal recover relative
        weights from any start node while keeping a single source of truth for
        the consistency checks.
        """

        def _register(a: str, b: str, delta: float, desc: str):
            key = (a, b)
            prev = directed_constraints.get(key)
            if prev is None:
                directed_constraints[key] = (delta, desc)
                adjacency.setdefault(a, []).append((b, delta, desc))
                return

            prev_delta, prev_desc = prev
            if not math.isclose(float(prev_delta), float(delta), abs_tol=tol):
                raise ValueError(f"Inconsistent pKa/tautomer constraints for {a}->{b}: {prev_desc} gives {prev_delta}, "
                                 f"but {desc} gives {delta}")

        _register(src, dst, float(delta_log10), source_desc)
        _register(dst, src, -float(delta_log10), source_desc)

    @staticmethod
    def _solve_log10_component_weights(
        nodes_list: list[str],
        adjacency: dict[str, list[tuple[str, float, str]]],
        *,
        tol: float = 1e-8,
    ) -> dict[str, float]:
        """Solve all relative state weights in one connected component.

        The generalized formula only needs relative populations, not absolute
        probabilities. We therefore choose an arbitrary root state with
        `log10(weight(root)) = 0` and propagate all other states via the stored
        pairwise constraints:

            log10(weight(j)) = log10(weight(i)) + delta_ij

        Any alternative path that reaches the same node must reproduce the same
        value within `tol`. If not, the user-provided `population` and/or
        `pKa_list` information is self-contradictory, so we raise immediately.

        The returned map is gauge-free up to a global constant shift. Downstream
        code only uses weight ratios `weight(i) / weight(ref)`, so the choice of
        root does not affect the final merged free energy.
        """

        node_set = set(nodes_list)
        root = nodes_list[0]
        log10_weights = {root: 0.0}
        stack = [root]

        while stack:
            node = stack.pop()
            base = float(log10_weights[node])
            for nxt, delta, source_desc in adjacency.get(node, []):
                if nxt not in node_set:
                    continue

                expected = base + float(delta)
                if nxt not in log10_weights:
                    log10_weights[nxt] = expected
                    stack.append(nxt)
                    continue

                if not math.isclose(float(log10_weights[nxt]), expected, abs_tol=tol):
                    raise ValueError(
                        f"Inconsistent pKa/tautomer component {nodes_list}: constraint {source_desc} expects "
                        f"log10_weight[{nxt}]={expected}, existing value is {log10_weights[nxt]}")

        missing = [node for node in nodes_list if node not in log10_weights]
        if missing:
            raise ValueError(f"Failed to solve pKa/tautomer component weights for nodes: {missing}")
        return log10_weights

    @staticmethod
    def _solve_component_charge_levels(
        nodes_list: list[str],
        pka_edges: set[tuple[str, str]],
        same_charge_edges: set[tuple[str, str]],
        *,
        max_charge_levels: int = 2,
    ) -> dict[str, int]:
        """Infer discrete charge levels inside one merged component.

        We support only three topology families for RBFE pKa/tautomer merging:

        1. all states share one charge state (pure tautomer/population case)
        2. one A- state shared by many HA states
        3. one HA state shared by many A- states

        To distinguish these cases we propagate an integer `charge_level` across
        the component:

        - population / tautomer edges imply `delta_level = 0`
        - pKa edges are directed `A- -> HA`, so they imply `delta_level = +1`

        The absolute number is arbitrary; only relative levels matter. After the
        DFS/BFS pass we normalize the minimum level to zero. The result is then
        used as a strict topology self-check before applying the generalized
        partition-function formula.

        Self-checks performed here
        --------------------------
        - a node cannot be assigned two incompatible charge levels
        - the component cannot span more than `max_charge_levels` charge values
          (currently 2: one deprotonated level and optionally one protonated
          level)
        """

        if not pka_edges:
            return {node: 0 for node in nodes_list}

        node_set = set(nodes_list)
        adjacency: dict[str, list[tuple[str, int]]] = {node: [] for node in nodes_list}
        for node1, node2 in same_charge_edges:
            if node1 not in node_set or node2 not in node_set:
                continue
            adjacency.setdefault(node1, []).append((node2, 0))
            adjacency.setdefault(node2, []).append((node1, 0))
        for deprot_node, prot_node in pka_edges:
            if deprot_node not in node_set or prot_node not in node_set:
                continue
            adjacency.setdefault(deprot_node, []).append((prot_node, 1))
            adjacency.setdefault(prot_node, []).append((deprot_node, -1))

        levels: dict[str, int] = {}
        for start in nodes_list:
            if start in levels:
                continue
            levels[start] = 0
            stack = [start]
            cur_pka_tautomer_connected_group = set()
            while stack:
                node = stack.pop()
                cur_pka_tautomer_connected_group.add(node)
                base_level = levels[node]
                for nxt, delta in adjacency.get(node, []):
                    expected = base_level + delta
                    if nxt not in levels:
                        levels[nxt] = expected
                        stack.append(nxt)
                        continue
                    if levels[nxt] != expected:
                        raise ValueError(
                            f"Inconsistent protonation levels in pKa/tautomer component {nodes_list}: "
                            f"node '{nxt}' has both level {levels[nxt]} and {expected}.\nLikely pKa_list pair sequence rule (deprot, prot, pKa) not followed."
                        )

            # for each connected group of nodes, we assign the lowest charge level to 0
            level_offset = min(levels[node] for node in cur_pka_tautomer_connected_group)
            for node in cur_pka_tautomer_connected_group:
                levels[node] -= level_offset
                if levels[node] >= max_charge_levels:
                    raise ValueError(
                        f"Only support up to {max_charge_levels} charge levels (A- and HA). Unsupported pKa/tautomer component {nodes_list}: node '{node}' has charge level {levels[node]} relative to min charge level 0."
                    )

        return levels

    @staticmethod
    def _classify_supported_component_topology(
        nodes_list: list[str],
        charge_levels: dict[str, int],
        *,
        has_pka: bool,
    ) -> str:
        """Classify one pka/tautomer connected component into the only three topologies we support.

        Returns
        -------
        `same_charge`
            No pKa edge is present. All states are treated as same-charge
            tautomers/conformers with externally supplied population ratios.

        `single_deprot_many_prot`
            Exactly one state is on the lower-charge side (A- / deprotonated) and
            one or more states are on the higher-charge side (HA / protonated).

        `many_deprot_single_prot`
            The symmetric case: multiple A- states share exactly one HA state.

        Anything outside these three families is rejected intentionally. In
        particular, we do NOT support a component containing multiple A- states
        and multiple HA states simultaneously, even if the pairwise ratios are
        mathematically solvable, because the current production scope is limited
        to the three user-specified situations.
        """
        if not has_pka:
            return "same_charge"

        deprot_states = [node for node in nodes_list if charge_levels[node] == 0]
        prot_states = [node for node in nodes_list if charge_levels[node] == 1]
        other_levels = sorted({charge_levels[node] for node in nodes_list} - {0, 1})
        if other_levels:
            raise ValueError(
                f"Unsupported pKa/tautomer component {nodes_list}: unexpected normalized charge levels {other_levels}.\nFull charge levels: {charge_levels}"
            )

        if not deprot_states or not prot_states:
            raise ValueError(
                f"Unsupported pKa/tautomer component {nodes_list}: pKa edges exist but one charge side is empty..\nFull charge levels: {charge_levels}"
            )

        if len(deprot_states) == 1:
            return "single_deprot_many_prot"
        if len(prot_states) == 1:
            return "many_deprot_single_prot"
        raise NotImplementedError(
            f"Unsupported pKa/tautomer topology for component {nodes_list}: found {len(deprot_states)} A- states and "
            f"{len(prot_states)} HA states. Supported cases are only: same-charge group, one A- with many HA, "
            f"or one HA with many A-.")

    def _merge_weighted_component(
        self,
        nodes_list: list[str],
        log10_weights: dict[str, float],
        ref_node: str,
        *,
        thermal_rt: float,
    ) -> tuple[float, float | None, list, list[float]]:
        """Merge one validated component with the generalized partition formula.

        After the topology checks, every supported case can be written in the
        same compact form. The ref_node is a state of A- form `ref` and let

            r_i = weight(i) / weight(ref)

        where `weight(i)` comes from the solved `population` and/or `pKa`
        constraints. The component free energy compared to the experimental
        mixture is then

            dG_merge = dG_ref - RT * ln(
                sum_i r_i * exp(-(dG_i - dG_ref) / RT)
                / sum_i r_i
            )

        This is exactly the form derived in the user's Lark document for:
        - same-charge tautomer mixing
        - one A- with many HA states
        - one HA with many A- states

        Self-checks here are intentionally lightweight:
        - all states in one component must carry the same experimental reference
          value if `exp` is present
        - `avg_dG_parts` is normalized to a list before aggregation so the merged
          node preserves provenance for downstream edge reconstruction and tests
        """

        exp_vals = [self.g.nodes[n].get("exp") for n in nodes_list]
        valid_exps = [e for e in exp_vals if e is not None]
        if valid_exps and len(set(valid_exps)) != 1:
            raise ValueError(
                f"Nodes in pKa/tautomer component {nodes_list} have differing experimental values: {set(valid_exps)}")
        exp_out = valid_exps[0] if valid_exps else None

        for n in nodes_list:
            self.g.nodes[n]["avg_dG_parts"] = [self.g.nodes[n]["dG"]]

        ref_dg = float(self.g.nodes[ref_node]["dG"])
        ref_log10_weight = float(log10_weights[ref_node])

        num_term = 0.0
        den_term = 0.0
        for node in nodes_list:
            ratio = 10**(float(log10_weights[node]) - ref_log10_weight)
            dg = float(self.g.nodes[node]["dG"])
            boltz_term = math.exp(-(dg - ref_dg) / thermal_rt)
            num_term += ratio * boltz_term
            den_term += ratio

        merged_dg = ref_dg - thermal_rt * math.log(num_term / den_term)

        merged_ligand_parts = []
        merged_avg_dg_parts = []
        for n in nodes_list:
            merged_ligand_parts.append(self.g.nodes[n]["ligand_parts"])
            merged_avg_dg_parts.extend(self.g.nodes[n]["avg_dG_parts"])

        return merged_dg, exp_out, merged_ligand_parts, merged_avg_dg_parts

    def pka_taut_correct(self, debug: bool = False):
        """
        Applies generalized pKa / tautomer correction on the current node graph.
        """
        merged_nodes = set()

        if 'pka_tautomer' in self.correction_config:
            pka_config = self.correction_config['pka_tautomer']
            orig_to_current = self._orig_ligand_to_current_node()
            tol = 1e-8
            component_graph = nx.Graph()
            adjacency: dict[str, list[tuple[str, float, str]]] = {}
            directed_constraints: dict[tuple[str, str], tuple[float, str]] = {}
            pka_pairs: set[tuple[str, str]] = set()
            directed_pka_edges: set[tuple[str, str]] = set()
            same_charge_edges: set[tuple[str, str]] = set()

            def resolve_current_node(orig_name: str, *, source_name: str) -> str:
                node = orig_to_current.get(str(orig_name))
                if node is None:
                    raise ValueError(f"{source_name} refers to '{orig_name}' but it is missing from the current graph.")
                return node

            population = pka_config.get("population") or {}
            if population:
                if not isinstance(population, dict):
                    raise ValueError("pka_tautomer.population must be a dict")

                for pair_key, weights in population.items():
                    states = [s for s in str(pair_key).split(":") if s]
                    if len(states) < 2:
                        raise ValueError(f"Invalid population key '{pair_key}'")
                    if not isinstance(weights, (list, tuple)) or len(weights) != len(states):
                        raise ValueError(f"Population weights length mismatch for '{pair_key}': {weights}")

                    w = [float(x) for x in weights]
                    if any(x < 0.0 or x > 1.0 for x in w):
                        raise ValueError(f"Population weights must be in [0,1] for '{pair_key}': {weights}")
                    if abs(sum(w) - 1.0) >= 1e-3:
                        raise ValueError(f"Population weights must sum to 1 for '{pair_key}': {weights}")

                    node_weights: dict[str, float] = {}
                    ordered_nodes: list[str] = []
                    for st, wi in zip(states, w):
                        cur = resolve_current_node(st, source_name="Population list")
                        assert cur not in node_weights, f"A single population list contains duplicate node '{cur}'"
                        ordered_nodes.append(cur)
                        node_weights[cur] = float(wi)

                    if any(node_weights[node] <= 0.0 for node in ordered_nodes):
                        raise ValueError(
                            f"Population weights must remain positive after mapping current nodes for '{pair_key}': {node_weights}"
                        )

                    anchor = ordered_nodes[0]
                    anchor_weight = float(node_weights[anchor])
                    component_graph.add_nodes_from(ordered_nodes)
                    for node in ordered_nodes[1:]:
                        component_graph.add_edge(anchor, node)
                        same_charge_edges.add(tuple(sorted((anchor, node))))
                        self._add_log_ratio_constraint(
                            adjacency,
                            directed_constraints,
                            anchor,
                            node,
                            math.log10(float(node_weights[node]) / anchor_weight),
                            f"population[{pair_key}]",
                            tol=tol,
                        )

            pka_list = pka_config.get("pKa_list") or []
            if pka_list:
                if "pH" not in pka_config:
                    raise ValueError("pka_tautomer config must contain pH when pKa_list is provided")

                pH = float(pka_config["pH"])
                if self.pH is None:
                    self.pH = pH
                assert self.pH == pH, f"pH mismatch: {self.pH} != {pH} in config"

                for deprot_orig, prot_orig, pka_val in pka_list:
                    deprot_node = resolve_current_node(str(deprot_orig), source_name="pKa list")
                    prot_node = resolve_current_node(str(prot_orig), source_name="pKa list")
                    if deprot_node == prot_node:
                        raise ValueError(
                            f"pKa pair '{deprot_orig}->{prot_orig}' is invalid: deport and prot have the same current node '{deprot_node}'"
                        )

                    component_graph.add_edge(deprot_node, prot_node)
                    pka_pairs.add(tuple(sorted((deprot_node, prot_node))))
                    directed_pka_edges.add((deprot_node, prot_node))
                    self._add_log_ratio_constraint(
                        adjacency,
                        directed_constraints,
                        deprot_node,
                        prot_node,
                        float(pka_val) - pH,
                        f"pKa_list[{deprot_orig},{prot_orig},{pka_val}]",
                        tol=tol,
                    )

            components = [sorted(list(comp)) for comp in nx.connected_components(component_graph) if len(comp) > 1]
            for nodes_list in components:
                comp_edges = component_graph.subgraph(nodes_list).edges()
                has_pka = any(tuple(sorted((u, v))) in pka_pairs for u, v in comp_edges)

                log10_weights = self._solve_log10_component_weights(nodes_list, adjacency, tol=tol)
                charge_levels = self._solve_component_charge_levels(nodes_list, directed_pka_edges, same_charge_edges)
                self._classify_supported_component_topology(nodes_list, charge_levels, has_pka=has_pka)

                ref_node = [n for n in nodes_list if charge_levels[n] == 0][0]
                merged_dg, exp_out, merged_ligand_parts, merged_avg_dg_parts = self._merge_weighted_component(
                    nodes_list,
                    log10_weights,
                    ref_node,
                    thermal_rt=self.pka_rt,
                )

                merged_name = "+".join(nodes_list)
                self.g.add_node(merged_name,
                                dG=merged_dg,
                                exp=exp_out,
                                ligand_parts=merged_ligand_parts,
                                avg_dG_parts=merged_avg_dg_parts)
                merged_nodes.add(merged_name)
                self.g.remove_nodes_from(nodes_list)

        # For nodes that didn't go through pKa correction, wrap their ligand_parts
        for n in self.g.nodes():
            if n not in merged_nodes:
                self.g.nodes[n]['ligand_parts'] = [self.g.nodes[n]['ligand_parts']]
                self.g.nodes[n]['avg_dG_parts'] = [self.g.nodes[n]['avg_dG_parts']]

        if debug:
            self._autosave('03_pka')

    def solvent_correct(self, debug: bool = False) -> None:
        """
        Corrects the dG of solvent nodes.
        """
        if 'solvent_correction' in self.correction_config:
            solvent_correction = self.correction_config['solvent_correction']
            assert 'pH' in solvent_correction, "Solvent correction must have a pH value."
            pH = solvent_correction['pH']
            if self.pH is None:
                self.pH = pH
            assert pH == self.pH, f"pH value in config ({pH}) does not match the pH value in the correction_config ({self.pH})."

            assert 'ligand_list' in solvent_correction, "Solvent correction must have a ligand_list."
            ligand_list = solvent_correction['ligand_list']
            for lig in ligand_list:
                assert len(
                    ligand_list[lig]
                ) == 2, f"Solvent correction for {lig} must have 2 elements (pKa, lig current form: HA or A-)."
                pKa, lig_form = ligand_list[lig]
                assert lig_form in ['HA', 'A-'], f"Solvent correction for {lig} must have a valid form (HA or A-)."

                ratio = 10**(pKa - self.pH)  # [LH+] / [LH]
                if lig_form == 'HA':
                    dg_corr = -self.pka_rt * math.log((1.0 + ratio) / ratio)
                else:
                    dg_corr = -self.pka_rt * math.log(1.0 + ratio)

                self.g.nodes[lig]['dG'] -= dg_corr

        if debug:
            self._autosave('04_solvent')

    def get_results(self) -> pd.DataFrame:
        """
        Returns a DataFrame of the current nodes in the graph.
        """
        records = []
        for node, attrs in self.g.nodes(data=True):
            record = {
                'ligand': node,
                'avg/dG': attrs['dG'],
                'ligand_parts': attrs['ligand_parts'],
                'avg/dG_parts': attrs['avg_dG_parts']
            }
            if 'exp' in attrs and attrs['exp'] is not None:
                record['exp'] = attrs['exp']
            records.append(record)
        return pd.DataFrame(records)

    def _autosave(self, step_name: str) -> str:
        """Helper to auto-save the current state to a TSV file."""
        df_out = self.get_results()

        if self.output_dir:
            base = os.path.basename(self.basename)
            out_path = os.path.join(self.output_dir, f"{base}_{step_name}.tsv")
        else:
            out_path = f"{self.basename}_{step_name}.tsv"

        df_out.to_csv(out_path, sep='\t', index=False)
        print(f"Saved {step_name} results to {out_path}")
        return out_path


class PostCorrections(BasePostCorrections):
    """Post-corrections workflow for ABFEP / nodewise inputs."""

    df: pd.DataFrame | None = None

    def __init__(self,
                 all_dgs_csv: str,
                 correction_config: Dict,
                 output_dir: str = None,
                 *,
                 rt: float = BasePostCorrections.RT,
                 pka_rt: float | None = None):
        super().__init__(all_dgs_csv, correction_config, output_dir, rt=rt, pka_rt=pka_rt)

        # Load CSV/TSV, sniffing separator
        self.df = pd.read_csv(all_dgs_csv, sep=None, engine='python', dtype={'ligand': str})

        # Verify columns exist
        if 'ligand' not in self.df.columns or 'avg/dG' not in self.df.columns:
            raise ValueError("Input dataframe must contain 'ligand' and 'avg/dG' columns.")

        # Verify exp values match between correction_config and tsv/csv if provided
        if 'exp_ref' in self.correction_config:
            if 'exp' not in self.df.columns:
                raise ValueError("Config provides 'exp_ref' but dataframe lacks 'exp' column.")
            for _, row in self.df.iterrows():
                lig = row['ligand']
                if lig in self.correction_config['exp_ref']:
                    if not math.isclose(row['exp'], self.correction_config['exp_ref'][lig], abs_tol=5e-3):
                        raise ValueError(f"Experimental value mismatch for ligand {lig}: "
                                         f"CSV({row['exp']}) vs JSON({self.correction_config['exp_ref'][lig]})")
                else:
                    raise ValueError(f"Ligand {lig} found in CSV but missing from 'exp_ref' in config.")

        # Build initial graph
        for _, row in self.df.iterrows():
            lig = row['ligand']
            dg = row['avg/dG']
            exp_val = row['exp'] if 'exp' in self.df.columns else None

            self.g.add_node(lig, dG=dg, exp=exp_val, ligand_parts=lig, avg_dG_parts=dg)

    def run(self, debug: bool = False) -> str:
        """
        Auto-run all post-corrections steps for ABFEP calculations.
        """
        self.symm_correct(debug)
        self.conf_correct(debug)
        self.pka_taut_correct(debug)
        self.solvent_correct(debug)
        final_result = self._autosave("final")
        return final_result
