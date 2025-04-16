# Updated MaterialBuilder class
from pymatgen.core.structure import Structure
import pormake as pm
pm.log.logger.setLevel(pm.log.logging.CRITICAL)
import os
import io
import tempfile
import random
from typing import Optional, List, Union, Dict, Any
from .masks import BaseMask, RMSDMask

class MaterialBuilder:
    def __init__(self, topology_string: str, mask: Optional[BaseMask] = None, include_edges: bool = True):
        """
        Initialize a MaterialBuilder for creating MOF structures.
        
        Args:
            topology_string: The topology identifier string.
            mask: A BaseMask instance to use for filtering valid building blocks.
                  If None, will create a default RMSDMask.
            include_edges: Whether to include edge building blocks in the structure.
        """
        self.database = pm.Database()
        self.builder = pm.Builder()
        self.topology = self.database.get_topo(topology_string)
        self.include_edges = include_edges
        
        # Initialize vocabulary
        self.token_vocabulary = self.database._get_bb_list()
        
        if include_edges:
            self.token_vocabulary.append('none')
        
        self.termination_token = "[TER]"
        
        # Set up the mask
        if mask is None:
            # Create default RMSDMask if none provided
            self.mask = RMSDMask().initialize(self.topology, self.token_vocabulary, include_edges)
        else:
            # Initialize the provided mask with our topology and vocabulary
            self.mask = mask.initialize(self.topology, self.token_vocabulary, include_edges)
        
        # Store topology details
        self.n_nodes = len(self.topology.unique_cn)
        
        if include_edges:
            self.n_slots = self.n_nodes + self.topology.n_edge_types
            self.unique_edge_types = self.topology.unique_edge_types
        else:
            self.n_slots = self.n_nodes
            self.unique_edge_types = []

    def random_sequence(self) -> List[str]:
        """Generate a random valid building block sequence for the topology"""
        sequence = []
        for i in range(0, self.n_slots):
            # Get valid options for current position using the mask
            valid_indices = self.mask.forward_actions_at_each_slot[i]
            allowed_building_blocks = [self.token_vocabulary[a] for a in valid_indices]
            
            if allowed_building_blocks:  # Ensure there are options available
                sequence.append(random.choice(allowed_building_blocks))
            else:
                raise ValueError(f"No valid building blocks for position {i}")
        
        return sequence

    def make_pormake_mof(self, sequence: List[str]):
        """Create a MOF using the pormake builder from a sequence of building blocks"""
        # Remove termination token if it's there
        if sequence[-1] == self.termination_token:
            sequence = sequence[:-1]
        
        # Split the sequence into nodes and edges
        edges = {}

        # We don't have any edges
        if len(sequence) == self.n_nodes:
            nodes = sequence
        # We have filled all the edge slots
        else:
            # Nodes are first in the sequence, then edges, then TER
            nodes = sequence[:self.n_nodes]
            edge_bb_names = sequence[self.n_nodes:]
            edge_connections_to_delete = []

            for i in range(len(self.unique_edge_types)):
                edge_connections = (self.unique_edge_types[i][0], self.unique_edge_types[i][1])
                edges[edge_connections] = edge_bb_names[i] 

                if edge_bb_names[i] == 'none':
                    edge_connections_to_delete.append(edge_connections)

            for item in edge_connections_to_delete:
                del edges[item]

        return self._make_pormake_mof(nodes, edges)

    def _make_pormake_mof(self, nodes: List[str], edges: Dict[tuple, str]):
        """Internal method to build a MOF with specific nodes and edges"""
        node_bbs = []

        for node in nodes:
            node_bbs.append(self.database.get_bb(node))

        for key in edges:
            edges[key] = self.database.get_building_block(edges[key])

        return self.builder.build_by_type(topology=self.topology, node_bbs=node_bbs, edge_bbs=edges)
    
    def make_cif(self, sequence: List[str]) -> str:
        """Generate a CIF string representation of the MOF from a sequence"""
        mof = self.make_pormake_mof(sequence)
        cif_string = self.mof_to_cif_string(mof)
        return cif_string
    
    def make_structure(self, sequence: List[str]) -> Structure:
        """Generate a pymatgen Structure of the MOF from a sequence"""
        cif_string = self.make_cif(sequence)
        return Structure.from_str(cif_string, fmt="cif")

    def mof_to_cif_string(self, mof) -> str:
        """
        Convert a pormake MOF object to a CIF format string
        
        Adapted from framework.write_cif in pormake
        Only writes symmetry information and atom coordinates
        """
        f = io.StringIO("")

        f.write("data_{}\n")

        f.write("_symmetry_space_group_name_H-M    P1\n")
        f.write("_symmetry_Int_Tables_number       1\n")
        f.write("_symmetry_cell_setting            triclinic\n")

        f.write("loop_\n")
        f.write("_symmetry_equiv_pos_as_xyz\n")
        f.write("'x, y, z'\n")

        a, b, c, alpha, beta, gamma = \
            mof.atoms.get_cell_lengths_and_angles()

        f.write("_cell_length_a     {:.3f}\n".format(a))
        f.write("_cell_length_b     {:.3f}\n".format(b))
        f.write("_cell_length_c     {:.3f}\n".format(c))
        f.write("_cell_angle_alpha  {:.3f}\n".format(alpha))
        f.write("_cell_angle_beta   {:.3f}\n".format(beta))
        f.write("_cell_angle_gamma  {:.3f}\n".format(gamma))

        f.write("loop_\n")
        f.write("_atom_site_label\n")
        f.write("_atom_site_type_symbol\n")
        f.write("_atom_site_fract_x\n")
        f.write("_atom_site_fract_y\n")
        f.write("_atom_site_fract_z\n")
        f.write("_atom_type_partial_charge\n")

        symbols = mof.atoms.symbols
        frac_coords = mof.atoms.get_scaled_positions()
        for i, (sym, pos) in enumerate(zip(symbols, frac_coords)):
            label = "{}{}".format(sym, i)
            f.write("{} {} {:.5f} {:.5f} {:.5f} 0.0\n".
                    format(label, sym, *pos))

        return f.getvalue()
