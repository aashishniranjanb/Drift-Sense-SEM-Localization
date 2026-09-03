import numpy as np

def cluster_replica_families(candidates: list, scale: float) -> list:
    """
    Groups candidates into spatial repetition families (replica families) based on
    DRAM/FinFET pitches.
    
    Assigns to each candidate:
      - family_id (int)
      - family_population (int)
      - family_score_variance (float)
      
    Returns:
        candidates (list): Candidates list updated with family features.
    """
    if len(candidates) == 0:
        return candidates
        
    pitches = np.array([32.0, 36.0, 48.0, 128.0]) / scale
    n = len(candidates)
    
    # Adjacency matrix for family graph
    adj = np.eye(n, dtype=bool)
    
    for i in range(n):
        for j in range(i + 1, n):
            dx = abs(candidates[i]["cx"] - candidates[j]["cx"])
            dy = abs(candidates[i]["cy"] - candidates[j]["cy"])
            
            x_match = any(abs(dx - p * round(dx / p)) < 2.0 and round(dx / p) > 0 for p in pitches)
            y_match = any(abs(dy - p * round(dy / p)) < 2.0 and round(dy / p) > 0 for p in pitches)
            
            # Connected if they align vertically or horizontally along periodic grids
            if (x_match and dy < 2.5) or (y_match and dx < 2.5):
                adj[i, j] = True
                adj[j, i] = True
                
    # Connected components search (BFS)
    visited = np.zeros(n, dtype=bool)
    family_id = 0
    families = {}
    
    for i in range(n):
        if not visited[i]:
            queue = [i]
            visited[i] = True
            comp = []
            
            head = 0
            while head < len(queue):
                u = queue[head]
                head += 1
                comp.append(u)
                for v in range(n):
                    if adj[u, v] and not visited[v]:
                        visited[v] = True
                        queue.append(v)
                        
            families[family_id] = comp
            family_id += 1
            
    # Calculate properties for each family and annotate candidates
    for fam_id, member_indices in families.items():
        member_scores = [candidates[idx]["corr_score"] for idx in member_indices]
        pop = len(member_indices)
        variance = float(np.var(member_scores)) if pop > 1 else 0.0
        
        for idx in member_indices:
            candidates[idx]["family_id"] = fam_id
            candidates[idx]["family_population"] = pop
            candidates[idx]["family_score_variance"] = variance
            
    return candidates
