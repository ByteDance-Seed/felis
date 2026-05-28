# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
from abc import ABC, abstractmethod
from collections import namedtuple
from pathlib import Path

import networkx as nx
import parmed
from networkx import Graph
from networkx.algorithms import isomorphism
from packaging import version
from parmed import read_PDB
from parmed.geometry import STANDARD_BOND_LENGTHS_SQUARED, distance2

from bytemol.toolkit.protein.helper.parse_residue_lib import get_amber_protein_res_template, get_ion_solvent_template

logger = logging.getLogger(__name__)

assert version.parse(parmed.__version__) >= version.parse('4.2')

NBOND_LIMITS = {'H': 1, 'C': 4, 'N': 4, 'O': 2, 'P': 4, 'S': 2}
RESTEMP_LIB = get_amber_protein_res_template()
ION_SOL_LIB = get_ion_solvent_template()
PROTEIN_RESIDUE_PROTONATED_STATES = {
    'HIS': ['HID', 'HIE', 'HIP'],
    'LYS': ['LYN'],
    'ASP': ['ASH'],
    'GLU': ['GLH'],
    'CYS': ['CYX', 'CYM']
}
PROTONATED_RES_TO_STANDARD = {
    p: s for s in PROTEIN_RESIDUE_PROTONATED_STATES.keys() for p in PROTEIN_RESIDUE_PROTONATED_STATES[s]
}


class BaseSanitizer(ABC):

    def _graph_single_pdbResidue(self, res: parmed.Residue):
        """
        Convert a single residue into a graph representation
        Parameter
        ---------
        res: Residue read from the pdb. Stored as parmed Residue class

        Return
        ------
        G: Graph representation of the residue. Stored are nx.Graph
        """
        G = nx.Graph()
        n_atoms_in_res = len(res.atoms)

        for i in range(n_atoms_in_res):
            atom1 = res.atoms[i]
            for j in range(i + 1, n_atoms_in_res):
                atom2 = res.atoms[j]
                maxdist = STANDARD_BOND_LENGTHS_SQUARED[(atom1.atomic_number, atom2.atomic_number)] * 1.1
                if distance2(atom1, atom2) < maxdist:
                    G.add_edge(atom1.number, atom2.number)

        atom_attr_mapping = dict()
        for a in res.atoms:
            atom_attr_mapping[a.number] = {'element': a.element_name, 'atomname': a.name}
        nx.set_node_attributes(G, atom_attr_mapping)

        if len(G.nodes) != n_atoms_in_res:
            raise RuntimeError(f'[PDBSantinizer] Atoms in residue {res.number}{res.name} are not fully connected!')

        return G

    def _generate_structure_type_info(self, query_residue: parmed.Residue, query_residue_G: Graph,
                                      matched_residue: str):
        res_sturctype_info = dict()
        # get mapping (atom_number -> atomname)
        target_graph = self.res_temp_lib[matched_residue]['resGraph']
        mapping = nx.vf2pp_isomorphism(query_residue_G, target_graph, node_label='element')
        for a in query_residue.atoms:
            atom_number = a.number
            atom_idx = a.idx + 1  # change to 1-based
            atom_name = mapping[atom_number]
            target_node = target_graph.nodes[atom_name]
            structure_type = target_node.get('structuretype')
            element = target_node.get('element')
            assert len(structure_type
                      ) > 0, f'wrong structure type {structure_type} for atom {atom_number} in {matched_residue}'
            # save as dict, later include improper info
            # use atom_idx as the key, so that the idx is the same as the idx in the gro
            res_sturctype_info[atom_idx] = {"element": element, "structure_type": structure_type}

        return res_sturctype_info

    @abstractmethod
    def get_full_conect(self):
        pass


class PureProteinSanitizer(BaseSanitizer):
    """
    Read a pdb file, process the structure and find out the connectivity within the protein structure. 
    Then add protein related CONECT records in to the output.
    """

    def __init__(self, pdb_file_path: str, pdb_connect_set: set, *, check_names: bool = False):
        self.pdb_file_path: str = str(Path(pdb_file_path).resolve())
        self.check_names: bool = check_names
        self.res_temp_lib: dict = RESTEMP_LIB
        if self.check_names:
            self._replace_names: dict = {}  # atom_serial_number -> (replace_res_name, replace element)

        # We currently use the Structure class in parmed.sturcture, will need to extract this out or rewrite our own
        self.protein_parmed_struc: parmed.Structure = read_PDB(self.pdb_file_path)
        self.protein_connect_set: set = pdb_connect_set
        self.all_atoms: set = set()
        self.all_bonds: set = set()

        self._structure_info: dict = {}  # atom_number => {atom_element: xx, structure_type: xx}

        # save S atoms from CYM residues to achieve disulfide bond detection
        self.cym_s_atoms = []

    ############
    # property
    ############

    @property
    def structure_info(self):
        return self._structure_info

    @property
    def replace_names(self):
        assert self.check_names
        return self._replace_names

    def _find_exact_matches(self, query_graph: Graph, res: parmed.Residue):
        """
        Find the matched residue template according to graph match result.
        Parameter
        ---------
        query_graph: the graph representation of an individual residue in the input pdb
        res_id: the id of residue in the parmed structure

        Return
        ------
        matching_resnames: a list which should contain only one matched residue template for the input query_graph
        """
        # List to hold the residue name of matching graphs
        matching_resnames = []

        # Iterate over the list of graphs
        for resname in self.res_temp_lib.keys():
            graph = self.res_temp_lib[resname]['resGraph']
            # Check if the current graph is an exact match according to the node attribute 'element' and the connectivity
            if nx.vf2pp_is_isomorphic(query_graph, graph, node_label='element') and len(graph.nodes) == len(
                    query_graph.nodes):
                # If it is, add the index to the list
                matching_resnames.append(resname)

        # Deal with the tempalte matching for CYX and CYM, which have the same residue structure but different ssbond and charge
        if len(matching_resnames) > 1:
            if set(matching_resnames) == set(['CYX', 'CYM']):
                found_SSBOND = False
                # Use a very specific way to determine CYX or CYM:
                # if the S atom in current CYS residue is connect to another S atom,
                # then it is considered as a SSBOND and the matched residue is CYX,
                # otherwise it is CYM
                S_in_query_res = [
                    node for node, node_attr in query_graph.nodes(data=True) if node_attr['element'] == 'S'
                ][0]
                connect_records_to_S = {bond for bond in self.protein_connect_set if S_in_query_res in bond}
                all_atoms_connect_to_S = {
                    atom for bond in connect_records_to_S for atom in bond if atom != S_in_query_res
                }
                for atom_serial in all_atoms_connect_to_S:
                    atom_in_parmed = next(
                        (atom for atom in self.protein_parmed_struc.atoms if atom.number == atom_serial), None)
                    if atom_in_parmed.element_name == 'S':
                        found_SSBOND = True
                if found_SSBOND:
                    matching_resnames = ['CYX']
                else:
                    matching_resnames = ['CYM']
            else:
                raise ValueError(
                    f"[PureProteinSanitizer] !!! More than one residue template matched for res {repr(res)} with matching_resnames {matching_resnames}!!!"
                )

        return matching_resnames

    def _replace_resname_and_atomname(self, query_residue: parmed.Residue, query_residue_G: Graph,
                                      matched_residue: str):
        # if matched residue is a residue with specific protonated state, replace this residue's name to standard residue name
        replace_res_name = None
        if matched_residue in PROTONATED_RES_TO_STANDARD:
            if query_residue.name == PROTONATED_RES_TO_STANDARD[matched_residue]:
                replace_res_name = query_residue.name
            else:
                logger.info(
                    f'[PureProteinSanitizer] Converting residue {query_residue.idx}{query_residue.name} to standard resname {PROTONATED_RES_TO_STANDARD[matched_residue]}'
                )
                replace_res_name = PROTONATED_RES_TO_STANDARD[matched_residue]
        else:
            replace_res_name = query_residue.name

        assert replace_res_name is not None
        # replace atomnames to reference atomnames
        mapping = nx.vf2pp_isomorphism(query_residue_G,
                                       self.res_temp_lib[matched_residue]['resGraph'],
                                       node_label='element')
        for a in query_residue.atoms:
            # a.name = mapping[a.number]
            self._replace_names[a.number] = (replace_res_name, mapping[a.number])

    def _get_head_or_tail(self, G1: Graph, G2: Graph, target_atom: str):

        matcher = isomorphism.GraphMatcher(G1, G2, node_match=lambda n1, n2: n1['element'] == n2['element'])
        if matcher.is_isomorphic():
            mapping = matcher.mapping
            reverse_mapping = {v: k for k, v in mapping.items()}
            try:
                corresponding_node_in_G1 = reverse_mapping[target_atom]
                return corresponding_node_in_G1
            except KeyError:
                return None
        else:
            raise RuntimeError

    def _assign_bonds(self):
        """
        Collect topology info of the protein structure in the input pdb file by determining intra- and inter-molecular connectivity:
        1. Check the topology of each protein residue by graph matching with template structures, and determine intramolecular connectivity
        2. Determine the head and tail atoms of each protein residue, which will be used for the determination of intermolecular connectivity
        
        Return
        ------
        assigned_bonds
            A set of tuples containing the connectivity of the protein structure. 
            Each tuple contains a pair of atom indices that are bonded with each other.

        Raises
        ------
        RuntimeError
            When an expected disulfide bond is undefined.
        RuntimeError
            When a protein residue cannot be matched with template structures.
        RuntimeError
            When a protein chain is found broken with a gap.
        RuntimeError
            When a protein chain is found uncapped or with improper terminal residues.
        """
        assigned_graph_info = namedtuple('assigned_graph_info', ['matched_res', "query_graph"])

        assigned_res = dict()  # res_id => (template_res_name, graph)
        assigned_bonds = set()
        assigned_res_head_tail = dict()

        # step1: find match res and assign bonds within the residue
        for resid in range(len(self.protein_parmed_struc.residues)):
            query_res = self.protein_parmed_struc.residues[resid]
            query_res_G = self._graph_single_pdbResidue(query_res)
            matched_residues = self._find_exact_matches(query_res_G, query_res)
            if matched_residues:
                for e in query_res_G.edges():
                    assigned_bonds.add(e)
                assigned_res[resid] = assigned_graph_info(matched_residues[0], query_res_G)

                # check SSBOND
                if matched_residues == ['CYM']:
                    s_atom = [at for at in query_res.atoms if at.atomic_number == 16]
                    if self.cym_s_atoms:
                        for at in self.cym_s_atoms:
                            ss_dist = distance2(s_atom[0], at)
                            # Threshold for SSBOND is set to be 3 Angstrom,
                            # same as the threshold in openmm: https://github.com/openmm/openmm/blob/51a112a336db6cc0aea1c32b90d9d0a1f262cbfd/wrappers/python/openmm/app/topology.py#L346
                            # When there are two S atoms from CYM residues closer than 3A, the input pdb will be rejected.
                            if ss_dist < 3.0**2:
                                raise RuntimeError(
                                    f'[PureProteinSanitizer] Distance between atom {at.number} and atom {s_atom[0].number} are closer than 3 Angstrom. There should be a disulfide bond between them!'
                                )
                    self.cym_s_atoms.extend(s_atom)

                # replace res_name and atom_name if needed
                if self.check_names:
                    self._replace_resname_and_atomname(query_res, query_res_G, matched_residues[0])

                # always generate structure info
                res_structure_info = self._generate_structure_type_info(query_res, query_res_G, matched_residues[0])
                self._structure_info.update(res_structure_info)
            else:
                logger.info('query_res %s', query_res)
                logger.info('query_res_G %s', query_res_G)
                raise RuntimeError(f'[PureProteinSanitizer] Cannot find matching standard residue template for'
                                   f' residue {query_res} with info {repr(query_res)}')

        # step2: assigned head and tail atoms according to the template residue
        for resid in assigned_res:
            curr_res_head = None
            curr_res_tail = None
            query_res_G = assigned_res[resid].query_graph
            template_res = self.res_temp_lib[assigned_res[resid].matched_res]
            template_res_G = template_res['resGraph']

            # Make sure not assigning tail for C-terminal residues, and not assigning head for N-terminal residues
            if "N" in template_res['TailOrHead']:
                curr_res_head = self._get_head_or_tail(query_res_G, template_res_G, 'N')

            if "C" in template_res['TailOrHead']:
                curr_res_tail = self._get_head_or_tail(query_res_G, template_res_G, 'C')
            assigned_res_head_tail[resid] = {'head': curr_res_head, 'tail': curr_res_tail}

        # step3: assign inter-residue connectivity, following the logic of N-to-C
        serial_to_atom = {atom.number: atom for atom in self.protein_parmed_struc.atoms}
        connected_head_and_tail = set()
        for resid, res in enumerate(self.protein_parmed_struc.residues[:-1]):
            next_res = self.protein_parmed_struc.residues[resid + 1]
            # if current residue is the last residue of the chain, will move on to next residue
            if res.ter:
                continue
            # if current residue and the next residue are in different chains, move on to the next
            if (res.chain and (res.chain != next_res.chain)):
                continue

            current_tail = assigned_res_head_tail[resid]['tail']
            next_head = assigned_res_head_tail[resid + 1]['head']
            if current_tail and next_head:
                tail_head_dist = distance2(serial_to_atom[current_tail], serial_to_atom[next_head])
                max_inter_res_dist = STANDARD_BOND_LENGTHS_SQUARED[(6, 7)]
                # Specifically using the distance of C-N bond as threshold to justify the inter-residue connectivity
                if tail_head_dist < max_inter_res_dist:
                    assigned_bonds.add((current_tail, next_head))
                    connected_head_and_tail.update([current_tail, next_head])
                else:
                    raise RuntimeError(
                        f'[PureProteinSanitizer] There is a gap between residue {repr(res)} and next residue {repr(next_res)}. Please check!'
                    )
        # Check if there are free head or tail atoms:
        # Usually, the tail atom of amino acid residue i should be connected with the head atom of amino acid residue i+1.
        # If a head/tail atom is not connected with another tail/head atom, it means this atom is missing one bond and the corresponding residue is uncapped.
        # This check is mainly for C- and N- terminal residues in protein structure.
        # In protein structure, each of the C- and N- terminal residues should only have one atom being either head or tail atom, instead of having both head and tail atoms at the same time.
        # If a terminal residue has both head and tail atoms, it will lead to an uncapped scenario and will be caught by the following check process.
        free_head_or_tail_atoms = []
        for resid, val in assigned_res_head_tail.items():
            if val['head'] and val['head'] not in connected_head_and_tail:
                free_head_or_tail_atoms.append(val['head'])
            if val['tail'] and val['tail'] not in connected_head_and_tail:
                free_head_or_tail_atoms.append(val['tail'])
        if free_head_or_tail_atoms:
            raise RuntimeError(
                f'[PureProteinSanitizer] Following atoms need to be capped: {", ".join(str(i) for i in free_head_or_tail_atoms)}. Please check your structure!'
            )
        return assigned_bonds

    def _any_extra_bonds(self, atoms: set, bonds: set):
        valid_bonds = {b for b in bonds if b[0] in atoms and b[1] in atoms}
        extra_bd = bonds - valid_bonds
        return extra_bd

    def _any_extra_atoms(self, atoms: set, bonds: set):
        G = nx.Graph()
        G.add_nodes_from(atoms)
        G.add_edges_from(bonds)
        extra_at = [atom for atom, degree in dict(G.degree()).items() if degree == 0]
        return extra_at

    def _valence_validation(self):
        G = nx.Graph()
        G.add_nodes_from(self.all_atoms)
        G.add_edges_from(self.all_bonds)
        atom_attr_mapping = dict()
        for a in self.protein_parmed_struc.atoms:
            atom_attr_mapping[a.number] = {'element': a.element_name}
        nx.set_node_attributes(G, atom_attr_mapping)

        suspicious_atoms = dict()
        for at, max_bd in NBOND_LIMITS.items():
            selected_atoms = [x for x, y in G.nodes(data=True) if y['element'] == at]
            atoms_with_more_bonds = [node for node, degree in G.degree(selected_atoms) if degree > max_bd]
            if atoms_with_more_bonds:
                suspicious_atoms[at] = atoms_with_more_bonds
        if suspicious_atoms:
            raise RuntimeError(f'Following atoms has more bonds than expected:\n{list(suspicious_atoms.values())}')

    def _cross_validation(self, merged_conect_info):
        self.all_bonds = set(tuple(sorted(b)) for b in merged_conect_info)
        self.all_atoms = set(atom.number for atom in self.protein_parmed_struc)

        extra_bonds = self._any_extra_bonds(self.all_atoms, self.all_bonds)
        extra_atoms = self._any_extra_atoms(self.all_atoms, self.all_bonds)
        if extra_bonds:
            raise RuntimeError(f'[PureProteinSanitizer] Extra bonds found: {extra_bonds}')
        if extra_atoms:
            raise RuntimeError(f'[PureProteinSanitizer] Extra atoms found: {extra_atoms}')

    def get_full_conect(self):

        # step1: assign bonds to get all connect info
        logger.info("[PureProteinSanitizer] start assign bond for standard residues...")
        bonds_assigned = self._assign_bonds()
        merged_conect_info = self.protein_connect_set.union(bonds_assigned)

        # step2: cross_validation
        logger.info("[PureProteinSanitizer] start final cross_validation...")
        self._cross_validation(merged_conect_info)
        self._valence_validation()
        logger.info("[PureProteinSanitizer] End")

        return self.all_bonds


class PureWaterSanitizer(BaseSanitizer):

    def __init__(self, water_pdb_path: str):
        self.water_pdb_path = str(Path(water_pdb_path).resolve())
        self.water_structure: parmed.Structure = read_PDB(self.water_pdb_path, skip_bonds=True)
        self.res_temp_lib: dict = ION_SOL_LIB
        self.tgt_res: str = 'HOH'
        self.tgt_G: Graph = self.res_temp_lib[self.tgt_res]['resGraph']

        self.all_bonds: set = set()
        self._structure_info: dict = {}

    @property
    def structure_info(self):
        return self._structure_info

    def _assign_bonds(self):

        assign_bonds = []
        for _, query_res in enumerate(self.water_structure.residues):
            query_res_G = self._graph_single_pdbResidue(query_res)

            if not (nx.vf2pp_is_isomorphic(query_res_G, self.tgt_G, node_label='element') and
                    len(self.tgt_G.nodes) == len(query_res_G.nodes)):
                raise RuntimeError(
                    f"Please double check PDB: water {[repr(atom) for atom in query_res.atoms]} cannot match with template"
                )

            for e in query_res_G.edges():
                assign_bonds.append(e)

            # always generate structure info
            res_structure_info = self._generate_structure_type_info(query_res, query_res_G, self.tgt_res)
            self._structure_info.update(res_structure_info)

        self.all_bonds = set(assign_bonds)

    def get_full_conect(self):

        # assign bond & structure info
        logger.info("[PureWaterSantizier] assign_bonds")
        self._assign_bonds()
        logger.info("[PureWaterSantizier] End")

        return self.all_bonds
