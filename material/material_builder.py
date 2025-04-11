# MOF
from pymatgen.core.structure import Structure
import pormake as pm
pm.log.logger.setLevel(pm.log.logging.CRITICAL)
import os

# Standard library
import io
import tempfile
import random
from typing import List, Dict, Optional, Any

class MaterialBuilder:
    """
    Advanced builder for reticular materials (MOFs, COFs, etc.).
    Handles topology, building block selection, and structure assembly.
    """
    def __init__(self, topology: str, block_rmsd_cutoff: float = 0.2, include_edges: bool = True):
        """
        Args:
            topology (str): Topology name or identifier.
            block_rmsd_cutoff (float): RMSD threshold for building block compatibility.
            include_edges (bool): Whether to include edge building blocks.
        """
        self.database = pm.Database()
        self.builder = pm.Builder()
        self.topology = self.database.get_topo(topology)
        self.include_edges = include_edges
        self.block_rmsd_cutoff = block_rmsd_cutoff
        self.token_vocabulary = self._init_token_vocabulary()
        self.termination_token = "[TER]"
        self.mask = RMSDMask(self.topology, self.token_vocabulary, block_rmsd_cutoff, include_edges)
        self.n_nodes = len(self.topology.unique_cn)
        self.n_slots = self.mask.n_slots
        self.unique_edge_types = getattr(self.topology, 'unique_edge_types', []) if include_edges else []

    def _init_token_vocabulary(self) -> List[str]:
        vocab = self.database._get_bb_list()
        if self.include_edges:
            vocab.append('none')
        return vocab

    def random_sequence(self) -> List[str]:
        """
        Generate a random valid sequence of building blocks for the topology.
        Returns:
            List[str]: Sequence of building block names.
        """
        sequence = []
        for i in range(self.n_slots):
            allowed = [self.token_vocabulary[a] for a in self.mask.forward_actions_at_each_slot[i]]
            sequence.append(random.choice(allowed))
        return sequence

    def build_structure(self, sequence: List[str]) -> Structure:
        """
        Build a pymatgen Structure from a sequence of building blocks.
        Args:
            sequence (List[str]): Sequence of building block names.
        Returns:
            Structure: pymatgen Structure object.
        """
        cif_string = self.make_cif(sequence)
        return Structure.from_str(cif_string, fmt="cif")

    def make_cif(self, sequence: List[str]) -> str:
        """
        Generate a CIF string for the assembled structure.
        Args:
            sequence (List[str]): Sequence of building block names.
        Returns:
            str: CIF format string.
        """
        mof = self._assemble_structure(sequence)
        return self._mof_to_cif_string(mof)

    def _assemble_structure(self, sequence: List[str]) -> Any:
        """
        Assemble the structure using the builder and database.
        Args:
            sequence (List[str]): Sequence of building block names.
        Returns:
            Any: Assembled structure object (from pormake).
        """
        # Remove termination token if present
        if sequence and sequence[-1] == self.termination_token:
            sequence = sequence[:-1]
        edges = {}
        if len(sequence) == self.n_nodes:
            nodes = sequence
        else:
            offset = len(self.unique_edge_types)
            nodes = sequence[:-offset]
            edge_bb_names = sequence[self.n_nodes:]
            edge_connections_to_delete = []
            for i, edge_type in enumerate(self.unique_edge_types):
                edge_connections = (edge_type[0], edge_type[1])
                edges[edge_connections] = edge_bb_names[i]
                if edge_bb_names[i] == 'none':
                    edge_connections_to_delete.append(edge_connections)
            for item in edge_connections_to_delete:
                del edges[item]
        node_bbs = [self.database.get_bb(node) for node in nodes]
        for key in edges:
            edges[key] = self.database.get_building_block(edges[key])
        return self.builder.build_by_type(topology=self.topology, node_bbs=node_bbs, edge_bbs=edges)

    def _mof_to_cif_string(self, mof: Any) -> str:
        """
        Convert a pormake MOF object to a minimal CIF string.
        Args:
            mof (Any): MOF object from pormake.
        Returns:
            str: CIF format string.
        """
        f = io.StringIO("")
        f.write("data_{}\n")
        f.write("_symmetry_space_group_name_H-M    P1\n")
        f.write("_symmetry_Int_Tables_number       1\n")
        f.write("_symmetry_cell_setting            triclinic\n")
        f.write("loop_\n")
        f.write("_symmetry_equiv_pos_as_xyz\n")
        f.write("'x, y, z'\n")
        a, b, c, alpha, beta, gamma = mof.atoms.get_cell_lengths_and_angles()
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
            label = f"{sym}{i}"
            f.write(f"{label} {sym} {pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f} 0.0\n")
        return f.getvalue()

class RMSDMask:
    """
    Mask for allowed building blocks at each topology slot, based on RMSD geometric similarity.
    Ensures only compatible building blocks are used, enforcing chemical plausibility.
    """
    def __init__(self, topology: Any, vocab: List[str], block_rmsd_cutoff: float, include_edges: bool):
        self.database = pm.Database()
        self.topology = topology
        self.vocab = vocab
        self.block_rmsd_cutoff = block_rmsd_cutoff
        self.include_edges = include_edges
        self.connection_points_at_each_position = self.topology.unique_cn
        self.n_nodes = len(self.connection_points_at_each_position)
        self.local_structures = self.topology.unique_local_structures
        self.locator = pm.Locator()
        self.n_slots = self.n_nodes + (self.topology.n_edge_types if include_edges else 0)
        self.edge_vocab = self._init_edge_vocab() if include_edges else []
        self.forward_actions_at_each_slot = self._get_forward_actions_for_all_slots()

    def _init_edge_vocab(self) -> List[int]:
        edge_vocab = []
        for i, name in enumerate(self.vocab):
            if name == 'none':
                edge_vocab.append(i)
            else:
                bb = self.database.get_building_block(name)
                if bb.n_connection_points == 2:
                    edge_vocab.append(i)
        return edge_vocab

    def _get_forward_actions_for_all_slots(self) -> List[List[int]]:
        """
        For each slot, compute allowed building blocks based on RMSD and connectivity.
        Returns:
            List[List[int]]: Allowed indices for each slot.
        """
        actions = []
        for position in range(self.n_nodes):
            allowed = []
            for i, name in enumerate(self.vocab):
                if name != 'none':
                    bb = self.database.get_building_block(name)
                    if bb.n_connection_points == self.connection_points_at_each_position[position]:
                        structure = self.local_structures[position]
                        rmsd = self.locator.calculate_rmsd(structure, bb)
                        # RMSD is a geometric similarity metric; lower means more similar
                        if rmsd < self.block_rmsd_cutoff:
                            allowed.append(i)
            actions.append(allowed)
        if self.include_edges:
            for _ in range(self.topology.n_edge_types):
                actions.append(self.edge_vocab)
        return actions

    def allowed_forward_actions(self, sequence: List[str]) -> List[int]:
        """
        Get allowed actions for the next slot, given the current sequence.
        Args:
            sequence (List[str]): Current sequence.
        Returns:
            List[int]: Allowed indices in vocab.
        """
        position = len(sequence)
        return self.forward_actions_at_each_slot[position]