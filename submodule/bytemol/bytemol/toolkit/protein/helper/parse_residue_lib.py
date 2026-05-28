# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

import logging
import xml.etree.ElementTree as ET
from functools import cache

import networkx as nx

from bytemol.utils import get_data_file_path

logger = logging.getLogger(__name__)


def process_xml_ff_file(xmlff_filename):

    ff_tree = ET.parse(xmlff_filename)
    ff_root = ff_tree.getroot()
    ff_res_lib = dict()

    for residue in ff_root.iter('Residue'):
        # Build a gragh for each tempalte residue according on atom connectivity.
        # node: atomname, node_attribute: element, edge: bond
        G = nx.Graph()
        atom_info = dict()
        tail_or_head = []
        res_name = residue.get('name')
        for at in residue.iter('Atom'):
            at_name = at.get('name')
            at_element = at.get('element')
            # temperorily, no strcture type info for nucleic acids
            at_structure_type = at.get('structuretype', "")
            atom_info[at_name] = {'element': at_element, 'atomname': at_name, "structuretype": at_structure_type}
        G.add_nodes_from(atom_info.keys())
        nx.set_node_attributes(G, atom_info)
        for bd in residue.iter('Bond'):
            bd_atomname1 = bd.get('atomName1')
            bd_atomname2 = bd.get('atomName2')
            G.add_edge(bd_atomname1, bd_atomname2)

        # read the ExternalBond records from the xml file to determine the atoms that can form inter-res bonds
        for eb in residue.iter('ExternalBond'):
            externalbond_atom = eb.get('atomName')
            tail_or_head.append(externalbond_atom)

        if nx.is_connected(G):
            ff_res_lib[res_name] = {'resGraph': G, 'TailOrHead': tail_or_head}
        else:
            raise RuntimeError(f'Found isolated atoms in residue {res_name}. Please check!')
    return ff_res_lib


@cache
def get_amber_protein_res_template():
    amber_res_lib = dict()
    merged_ff_xml = get_data_file_path('residue_reference/residue_template_lib.xml', 'bytemol.toolkit.protein')

    amber_res_lib = process_xml_ff_file(merged_ff_xml)

    return amber_res_lib


@cache
def get_ion_solvent_template():
    ion_solvent_lib = dict()
    ion_solvent_xml = get_data_file_path('residue_reference/ion_solvent_template_lib.xml', 'bytemol.toolkit.protein')

    ion_solvent_lib = process_xml_ff_file(ion_solvent_xml)
    return ion_solvent_lib
