import numpy as np
import logging
import mdtraj as md
from typing import Dict

logger = logging.getLogger("md_engine")

class InteractionAnalyzer:
    """Analyzes intermolecular interactions and aggregation in mixtures."""
    
    def analyze(self, trajectory_path: str, topology_path: str) -> Dict:
        """
        Compute aggregation index and contact frequencies.
        """
        logger.info("Analyzing intermolecular interactions...")
        traj = md.load(trajectory_path, top=topology_path)
        
        # Identify solute residues (exclude water/ions)
        solute_indices = traj.topology.select("not resname HOH and not resname WAT and not resname Cl and not resname Na")
        
        if len(solute_indices) == 0:
            return {"aggregation_index": 0.0, "largest_cluster": 0, "status": "No solutes found"}

        # Get number of solute molecules (residues)
        solute_residues = [r for r in traj.topology.residues if r.name not in ['HOH', 'WAT', 'Cl', 'Na']]
        n_solutes = len(solute_residues)
        
        if n_solutes <= 1:
            return {"aggregation_index": 0.0, "largest_cluster": 1, "status": "Single solute"}

        # Compute pairwise distances between center of masses of solutes
        # For simplicity, let's use the first atom of each solute as a proxy or compute COM
        coms = []
        for res in solute_residues:
            atom_indices = [a.index for a in res.atoms]
            coms.append(md.compute_center_of_mass(traj.atom_slice(atom_indices)))
            
        coms = np.array(coms) # (n_frames, n_solutes, 3)
        
        # Analyze last 20 frames for aggregation
        last_frames = coms[-20:]
        avg_aggregation_index = 0.0
        max_cluster_size = 1
        
        # Threshold for contact (0.5 nm)
        threshold = 0.5 
        
        for frame_idx in range(len(last_frames)):
            frame_coms = last_frames[frame_idx]
            # Simple adjacency matrix
            adj = np.zeros((n_solutes, n_solutes))
            for i in range(n_solutes):
                for j in range(i + 1, n_solutes):
                    dist = np.linalg.norm(frame_coms[i] - frame_coms[j])
                    if dist < threshold:
                        adj[i, j] = adj[j, i] = 1
            
            # Find clusters (connected components)
            clusters = self._find_clusters(adj)
            if clusters:
                largest = max(len(c) for c in clusters)
                max_cluster_size = max(max_cluster_size, largest)
                avg_aggregation_index += largest / n_solutes
                
        avg_aggregation_index /= len(last_frames)
        
        return {
            "aggregation_index": round(float(avg_aggregation_index), 3),
            "largest_cluster": int(max_cluster_size),
            "contact_frequency": round(float(np.mean(adj)), 4),
            "total_solutes": n_solutes
        }

    def _find_clusters(self, adj):
        """Standard DFS to find connected components."""
        visited = set()
        clusters = []
        n = adj.shape[0]
        for i in range(n):
            if i not in visited:
                comp = []
                stack = [i]
                visited.add(i)
                while stack:
                    u = stack.pop()
                    comp.append(u)
                    for v in range(n):
                        if adj[u, v] == 1 and v not in visited:
                            visited.add(v)
                            stack.append(v)
                clusters.append(comp)
        return clusters
