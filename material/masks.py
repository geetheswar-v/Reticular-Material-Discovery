# Abstract Mask Base Class
import abc
import pormake as pm
pm.log.logger.setLevel(pm.log.logging.CRITICAL)
import numpy as np
from typing import List, Any, Dict, Tuple, Optional, Union

class BaseMask(abc.ABC):
    """
    Abstract base class for masks that filter building blocks based on 
    compatibility with topology positions.
    """
    def __init__(self):
        self.database = pm.Database()
        self.topology = None
        self.vocab = None
        self.n_nodes = 0
        self.n_slots = 0
        self.include_edges = False
        self.edge_vocab = []
        self.forward_actions_at_each_slot = []
    
    def initialize(self, topology: Any, vocab: List[str], include_edges: bool = True):
        """
        Initialize the mask with topology, vocabulary, and edge configuration.
        This is called by MaterialBuilder when the mask is attached.
        """
        self.topology = topology
        self.vocab = vocab
        self.include_edges = include_edges
        
        # Node counts and slots
        self.n_nodes = len(self.topology.unique_cn)
        
        # Calculate total slots (nodes + edges)
        if include_edges:
            self.n_slots = self.n_nodes + self.topology.n_edge_types
            # Identify which vocab entries can be used as edges
            self._identify_edge_vocab()
        else:
            self.n_slots = self.n_nodes
        
        # Compute allowed actions for each slot
        self.forward_actions_at_each_slot = self._compute_forward_actions()
        
        return self
    
    def _identify_edge_vocab(self):
        """Identify vocabulary items that can be used as edges (2 connection points)."""
        self.edge_vocab = []
        for i, name in enumerate(self.vocab):
            if name == 'none':
                self.edge_vocab.append(i)
            else:
                bb = self.database.get_building_block(name)
                if bb.n_connection_points == 2:
                    self.edge_vocab.append(i)
    
    @abc.abstractmethod
    def _compute_forward_actions(self) -> List[List[int]]:
        """
        Compute allowed actions for each slot based on mask criteria.
        Must be implemented by derived classes.
        """
        pass
    
    def allowed_forward_actions(self, sequence: List[str]) -> List[int]:
        """Return allowed actions at the current step in the sequence."""
        return self.forward_actions_at_each_slot[len(sequence)]


class RMSDMask(BaseMask):
    """
    Mask that filters building blocks based on RMSD (Root Mean Square Deviation)
    between the building block and the local structure in the topology.
    """
    def __init__(self, rmsd_cutoff: float = 0.3):
        super().__init__()
        self.rmsd_cutoff = rmsd_cutoff
        self.locator = pm.Locator()
    
    def _compute_forward_actions(self) -> List[List[int]]:
        """Compute allowed actions for each slot based on RMSD constraints."""
        actions = []
        
        # Process node slots
        for position in range(self.n_nodes):
            allowed_actions = []
            conn_points = self.topology.unique_cn[position]
            local_structure = self.topology.unique_local_structures[position]
            
            for i, name in enumerate(self.vocab):
                if name == 'none':
                    continue
                
                bb = self.database.get_building_block(name)
                
                # Check connection points match
                if bb.n_connection_points == conn_points:
                    # Calculate RMSD
                    rmsd = self.locator.calculate_rmsd(local_structure, bb)
                    
                    if rmsd < self.rmsd_cutoff:
                        allowed_actions.append(i)
            
            actions.append(allowed_actions)
        
        # Add edge slots if needed
        if self.include_edges:
            for _ in range(self.topology.n_edge_types):
                actions.append(self.edge_vocab)
        
        return actions


class AngleMask(BaseMask):
    """
    Mask that filters building blocks based on angular deviations 
    between the connection points in the building block and in the topology.
    """
    def __init__(self, angle_cutoff: float = 5.0):
        super().__init__()
        self.angle_cutoff = angle_cutoff  # in degrees
    
    def _angles_from_positions(self, pos: np.ndarray) -> np.ndarray:
        """
        Given an array of shape (k,3) of normalized connection-point vectors,
        return the flattened list of angles (in degrees) between each pair.
        """
        k = pos.shape[0]
        angles = []
        for i in range(k):
            for j in range(i+1, k):
                v1 = pos[i] / np.linalg.norm(pos[i])
                v2 = pos[j] / np.linalg.norm(pos[j])
                cosθ = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angles.append(np.degrees(np.arccos(cosθ)))
        return np.array(angles)
    
    def _compute_forward_actions(self) -> List[List[int]]:
        """Compute allowed actions for each slot based on angular constraints."""
        actions = []
        
        # Process node slots
        for slot_idx in range(self.n_nodes):
            allowed = []
            # Reference angles for this node
            ref_struct = self.topology.unique_local_structures[slot_idx]
            ref_pos = ref_struct.positions
            ref_angles = self._angles_from_positions(ref_pos)
            
            for i, name in enumerate(self.vocab):
                if name == 'none':
                    continue
                
                bb = self.database.get_building_block(name)
                # Must have the correct number of connection points
                if bb.n_connection_points != self.topology.unique_cn[slot_idx]:
                    continue
                
                bb_pos = bb.local_structure().positions
                bb_angles = self._angles_from_positions(bb_pos)
                
                # Compute RMS of angle differences
                diff = np.sqrt(np.mean((ref_angles - bb_angles)**2))
                if diff < self.angle_cutoff:
                    allowed.append(i)
            
            actions.append(allowed)
        
        # Add edge slots if needed
        if self.include_edges:
            for _ in range(self.topology.n_edge_types):
                actions.append(self.edge_vocab)
        
        return actions


class CombinedMask(BaseMask):
    """
    Mask that only allows building blocks whose RMSD to the reference local structure
    is below `rmsd_cutoff` AND whose inter-connection-point angular deviation
    is below `angle_cutoff`.
    """
    def __init__(self, rmsd_cutoff: float = 0.2, angle_cutoff: float = 5.0):
        super().__init__()
        self.rmsd_cutoff = rmsd_cutoff
        self.angle_cutoff = angle_cutoff  # in degrees
        self.locator = pm.Locator()
    
    def _angles_from_positions(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute all pairwise angles (in degrees) between normalized connection vectors.
        positions: array of shape (k,3)
        """
        k = positions.shape[0]
        angles = []
        for i in range(k):
            v1 = positions[i] / np.linalg.norm(positions[i])
            for j in range(i+1, k):
                v2 = positions[j] / np.linalg.norm(positions[j])
                cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
                angles.append(np.degrees(np.arccos(cos_theta)))
        return np.array(angles)
    
    def _compute_forward_actions(self) -> List[List[int]]:
        """Compute allowed actions using both RMSD and angle constraints."""
        actions = []
        
        # Process node slots with both constraints
        for pos_idx in range(self.n_nodes):
            allowed = []
            ref_struct = self.topology.unique_local_structures[pos_idx]
            
            # Precompute reference angles
            ref_pos = ref_struct.get_sites_coordinates() if hasattr(ref_struct, 'get_sites_coordinates') else ref_struct.positions
            ref_angles = self._angles_from_positions(np.array(ref_pos))
            
            for idx, name in enumerate(self.vocab):
                if name == 'none':
                    continue
                    
                bb = self.database.get_building_block(name)
                # Must match connection-point count
                if bb.n_connection_points != self.topology.unique_cn[pos_idx]:
                    continue
                
                # RMSD check
                rmsd_val = self.locator.calculate_rmsd(ref_struct, bb)
                if rmsd_val > self.rmsd_cutoff:
                    continue
                
                # Angle check
                bb_local = bb.local_structure()
                bb_pos = bb_local.get_sites_coordinates() if hasattr(bb_local, 'get_sites_coordinates') else bb_local.positions
                bb_angles = self._angles_from_positions(np.array(bb_pos))
                angle_rms = np.sqrt(np.mean((ref_angles - bb_angles) ** 2))
                if angle_rms > self.angle_cutoff:
                    continue
                
                allowed.append(idx)
            
            actions.append(allowed)
        
        # Add edge slots if needed
        if self.include_edges:
            for _ in range(self.topology.n_edge_types):
                actions.append(self.edge_vocab.copy())
        
        return actions