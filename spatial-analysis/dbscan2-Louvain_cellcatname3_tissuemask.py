import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import gc
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
from collections import Counter
from scipy.stats import chi2_contingency, zscore, kruskal, mannwhitneyu
from scipy.cluster.hierarchy import linkage, dendrogram
from statsmodels.stats.multitest import multipletests
from itertools import combinations
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import networkx as nx
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# UPDATED: Consistent dataset colors as requested
DATASET_COLORS = {
    'IgG4': [0, 0, 1],       # Blue
    'AAV': [1, 0, 1],        # Magenta
    'NK2': [0, 1, 0],        # Green
}

def get_dataset_color(dataset):
    """Get consistent color for dataset."""
    return DATASET_COLORS.get(dataset, [0.5, 0.5, 0.5])  # Default gray

# Function to check and provide installation instructions
def check_dependencies():
    """Check if required packages are installed and provide installation instructions."""
    missing_packages = []
    louvain_available = False
    
    # Check NetworkX
    try:
        import networkx
        print("✓ NetworkX is available")
    except ImportError:
        missing_packages.append("networkx")
        print("✗ NetworkX is missing")
    
    # Check for Louvain packages
    try:
        from community import community_louvain
        print("✓ python-louvain is available")
        louvain_available = True
    except ImportError:
        try:
            import community as community_louvain
            print("✓ community package is available")
            louvain_available = True
        except ImportError:
            print("⚠ python-louvain not found, will use NetworkX fallback")
            try:
                from networkx.algorithms import community as nx_community
                print("✓ NetworkX community detection is available as fallback")
            except ImportError:
                missing_packages.append("networkx[default]")
                print("✗ NetworkX community detection not available")
    
    if missing_packages:
        print("\n" + "="*60)
        print("MISSING DEPENDENCIES DETECTED!")
        print("="*60)
        print("Please install the following packages:")
        for pkg in missing_packages:
            print(f"  pip install {pkg}")
        print("\nFor optimal performance, also install:")
        print("  pip install python-louvain")
        print("="*60)
        
        response = input("\nContinue anyway with available packages? (y/n): ").strip().lower()
        if response != 'y':
            print("Exiting. Please install the required packages and try again.")
            exit(1)
    
    return louvain_available

# Check dependencies at startup
LOUVAIN_AVAILABLE = check_dependencies()

# Import the appropriate community detection modules
if LOUVAIN_AVAILABLE:
    try:
        from community import community_louvain
        print("→ Using python-louvain for optimal community detection")
    except ImportError:
        import community as community_louvain
        print("→ Using community package for community detection")
else:
    from networkx.algorithms import community as nx_community
    print("→ Using NetworkX community detection (fallback)")
    print("  For better results, install: pip install python-louvain")

# Define a comprehensive list of immune cell types to focus on
IMMUNE_CELL_TYPES = [
    # T cells
    'NaiveCD4T', 'OtherCD4T', 'CD4Treg', 'Th1', 'MKI+CD4T', 
    'CD4Tcm-Tem', 'CD4+iTreg', 'Th1-EM', 'CD4Trm', 'ICOS+Treg', 
    'Th1-RM', 'CD103+CD4T', 'CD103+CD4Treg','MKI+CD103+CD4T', 'HLAII+CD27+CD4T', 
    'HLAII+CD27+Treg', 'HLAII+CD27+Th1-like', 'MKI+HLAII+CD27+CD4T', 
    'Tfh', 'Tfr', 'Tfh1', 'MKI+Tfh', 'TIGIT+Tfh', 'TIGIT+Tfr', 
    'TIGIT+Tfh1', 'MKI+TIGIT+Tfh', 'HLAII+GZMK+Tfh', 'HLAII+GZMK+Tfr', 
    'HLAII+GZMK+Tfh1', 'MKI+HLAII+GZMK+Tfh', 'GZMA+GZMK+CD4CTL', 
    'GZMA+GZMK+CD4Treg', 'GZMA+GZMK+Th1-EM', 'MKI+GZMA+GZMK+CD4CTL', 
    'GZMB+CD4CTL', 'GZMB+CD4Treg', 'GZMB+Th1-EM', 'MKI+GZMB+CD4CTL',
    'NaiveCD8T', 'otherCD8T', 'CD8Treg', 'Tc1', 'MKI+CD8T', 
    'GZMA+GZMK+CD8CTL', 'GZMA+GZMK+CD8Treg', 'GZMA+GZMK+Tc1-EM', 
    'GZMB+CD8CTL', 'GZMB+CD8Treg', 'GZMB+Tc1-EM', 'CD103+CD8Treg', 
    'CD103+Tc1-like', 'CD8Trm', 'CD69+CD8Treg', 'Tbet+CD8Trm', 
    'CD103+CD8Trm', 'CD8Trm-Treg', 'Tbet+CD103+CD8Trm', 'TIGIT+CD8T', 
    'TIGIT+CD8Treg', 'TIGIT+Tc1', 'CD8Tcm-Tem', 'CD8+iTreg', 'Tc1-EM',
    
    # Other lymphoid cells
    'γδT', 'γδTreg', 'γδT1', 'MKI+γδT', 'ILC', 'ImmatureNK', 
    'MatureNK', 'NKT',
    
    # B cells
    'NaiveB', 'DN3B', 'MKI+B', 'TransitionalB', 'otherB', 'Breg', 
    'GCB', 'Tbet+MemoryB', 'MemoryB', 'CD11c+B', 'DN2B', 'CD11c+CD27+B', 
    'AgeAsscB', 'Plasmablast', 'IgG4+Plasmablast', 'Plasma', 'IgG4+Plasma',
    
    # Myeloid cells
    'M1Macrophage', 'MPO+Macrophage', 'M2Macrophage', 'MerTK+M1Macrophage',
    'MerTK+M2Macrophage', 'GPNMB+M1Macrophage', 'GPNMB+M2Macrophage',
    'cDC1', 'cDC2', 'pDC', 'Neutrophil',

    'Myofibroblast-SmoothMuscle', 'CD90+Myofibroblast', 'MMT', 'Apoptotic'
]

# Define structural/non-immune cell types to exclude
STRUCTURAL_CELL_TYPES = [
    'ProximalTubular', 'DistalTubular', 'StressedTubular', 
    'Endothelial', 'LymphaticEndothelial', 'Podocyte', 'PEC',
    'RBC', 'TubuleBackground', 'unclassified'
]

# Define cell type order based on cellcat3_colors
CELLCAT3_ORDER = [
    # CD4+ T Cells (Blue spectrum)
    'NaiveCD4T', 'OtherCD4T', 'CD4Treg', 'Th1', 'MKI+CD4T', 
    'CD4Tcm-Tem', 'CD4+iTreg', 'Th1-EM', 'CD4Trm', 'ICOS+Treg', 
    'Th1-RM', 'CD103+CD4T', 'CD103+CD4Treg', 'MKI+CD103+CD4T','HLAII+CD27+CD4T', 
    'HLAII+CD27+Treg', 'HLAII+CD27+Th1-like', 'MKI+HLAII+CD27+CD4T', 
    'Tfh', 'Tfr', 'Tfh1', 'MKI+Tfh', 'TIGIT+Tfh', 'TIGIT+Tfr', 
    'TIGIT+Tfh1', 'MKI+TIGIT+Tfh', 'HLAII+GZMK+Tfh', 'HLAII+GZMK+Tfr', 
    'HLAII+GZMK+Tfh1', 'MKI+HLAII+GZMK+Tfh', 'GZMA+GZMK+CD4CTL', 
    'GZMA+GZMK+CD4Treg', 'GZMA+GZMK+Th1-EM', 'MKI+GZMA+GZMK+CD4CTL', 
    'GZMB+CD4CTL', 'GZMB+CD4Treg', 'GZMB+Th1-EM', 'MKI+GZMB+CD4CTL',
    
    # CD8+ T Cells (Red spectrum)
    'NaiveCD8T', 'otherCD8T', 'CD8Treg', 'Tc1', 'MKI+CD8T', 
    'GZMA+GZMK+CD8CTL', 'GZMA+GZMK+CD8Treg', 'GZMA+GZMK+Tc1-EM', 
    'GZMB+CD8CTL', 'GZMB+CD8Treg', 'GZMB+Tc1-EM', 'CD103+CD8Treg', 
    'CD103+Tc1-like', 'CD8Trm', 'CD69+CD8Treg', 'Tbet+CD8Trm', 
    'CD103+CD8Trm', 'CD8Trm-Treg', 'Tbet+CD103+CD8Trm', 'TIGIT+CD8T', 
    'TIGIT+CD8Treg', 'TIGIT+Tc1', 'CD8Tcm-Tem', 'CD8+iTreg', 'Tc1-EM',
    
    # Other Lymphoid cells (Purple spectrum)
    'γδT', 'γδTreg', 'γδT1', 'MKI+γδT', 'ILC', 'ImmatureNK', 
    'MatureNK', 'NKT',
    
    # B cells & Plasma cells (Green spectrum)
    'NaiveB', 'DN3B', 'MKI+B', 'TransitionalB', 'otherB', 'Breg', 
    'GCB', 'Tbet+MemoryB', 'MemoryB', 'CD11c+B', 'DN2B', 'CD11c+CD27+B', 
    'AgeAsscB', 'Plasmablast', 'IgG4+Plasmablast', 'Plasma', 'IgG4+Plasma',
    
    # Myeloid cells (Yellow/Orange/Brown spectrum)
    'M1Macrophage', 'MPO+Macrophage', 'M2Macrophage', 'MerTK+M1Macrophage',
    'MerTK+M2Macrophage', 'GPNMB+M1Macrophage', 'GPNMB+M2Macrophage',
    'cDC1', 'cDC2', 'pDC', 'Neutrophil',
    
    # Stromal cells (Pink shades)
    'Myofibroblast-SmoothMuscle', 'CD90+Myofibroblast', 'MMT',
    
    # Other/Misc
    'Apoptotic'
]

# Global color mapping for consistent CN colors across datasets and areas
GLOBAL_CN_COLORS = {}

def get_consistent_cn_color(global_cn, renumbering_map=None, excluded_cns=None):
    """Get a consistent color for a global CN across all visualizations."""
    # If this is an excluded CN, return light gray
    if excluded_cns and global_cn in excluded_cns:
        return 'lightgray'
    
    # Use renumbered CN for color consistency if available
    display_cn = renumbering_map.get(global_cn, global_cn) if renumbering_map else global_cn
    
    if display_cn not in GLOBAL_CN_COLORS:
        # Use a combination of qualitative colormaps for better color distinction
        all_colors = (list(plt.cm.tab20.colors) + 
                     list(plt.cm.tab20b.colors) + 
                     list(plt.cm.tab20c.colors) + 
                     list(plt.cm.Set3.colors) + 
                     list(plt.cm.Paired.colors))
        
        # Assign color based on CN number
        if display_cn == -1:
            GLOBAL_CN_COLORS[display_cn] = 'lightgray'
        else:
            color_idx = int(abs(display_cn)) % len(all_colors)
            GLOBAL_CN_COLORS[display_cn] = all_colors[color_idx]
    
    return GLOBAL_CN_COLORS[display_cn]

# MODIFIED: Filter structural/non-immune cells BEFORE DBSCAN
def filter_immune_cells_only(dfInfo, cell_type_col='CellCatName3'):
    """
    Filter to keep only immune cells before DBSCAN clustering.
    """
    print(f"  Original cell count: {len(dfInfo):,}")
    
    # Count original cell types
    original_counts = dfInfo[cell_type_col].value_counts()
    print(f"  Original cell types: {len(original_counts)}")
    
    # Filter to keep only immune cells
    immune_mask = dfInfo[cell_type_col].isin(IMMUNE_CELL_TYPES)
    dfInfo_immune = dfInfo[immune_mask].copy()
    
    print(f"  After immune filtering: {len(dfInfo_immune):,} cells ({len(dfInfo_immune)/len(dfInfo)*100:.1f}% retained)")
    
    # Count filtered cell types
    filtered_counts = dfInfo_immune[cell_type_col].value_counts()
    print(f"  Immune cell types retained: {len(filtered_counts)}")
    
    # Show which structural cell types were removed
    structural_in_data = [ct for ct in original_counts.index if ct in STRUCTURAL_CELL_TYPES]
    if structural_in_data:
        structural_removed_count = sum(original_counts[ct] for ct in structural_in_data)
        print(f"  Structural cells removed: {structural_removed_count:,} ({structural_removed_count/len(dfInfo)*100:.1f}%)")
        print(f"  Structural types removed: {structural_in_data}")
    
    # Show which other cell types were removed (not in IMMUNE_CELL_TYPES but also not in STRUCTURAL_CELL_TYPES)
    other_removed = [ct for ct in original_counts.index 
                    if ct not in IMMUNE_CELL_TYPES and ct not in STRUCTURAL_CELL_TYPES]
    if other_removed:
        other_removed_count = sum(original_counts[ct] for ct in other_removed)
        print(f"  Other cells removed: {other_removed_count:,} ({other_removed_count/len(dfInfo)*100:.1f}%)")
        print(f"  Other types removed: {other_removed}")
    
    return dfInfo_immune

# NEW: Function to create similarity graph for Louvain clustering
def create_similarity_graph(data, similarity_threshold=0.3, similarity_metric='cosine'):
    """
    Create a similarity graph from composition data for Louvain clustering.
    
    Parameters:
    - data: Feature matrix (n_samples x n_features)
    - similarity_threshold: Minimum similarity to create an edge
    - similarity_metric: 'cosine', 'correlation', or 'euclidean'
    
    Returns:
    - G: NetworkX graph
    - similarity_matrix: Full similarity matrix
    """
    print(f"Creating similarity graph with {similarity_metric} similarity...")
    print(f"Data shape: {data.shape}")
    
    # Calculate similarity matrix
    if similarity_metric == 'cosine':
        similarity_matrix = cosine_similarity(data)
    elif similarity_metric == 'correlation':
        similarity_matrix = np.corrcoef(data)
        similarity_matrix = np.nan_to_num(similarity_matrix)  # Handle NaN values
    elif similarity_metric == 'euclidean':
        # Convert euclidean distance to similarity
        distance_matrix = pairwise_distances(data, metric='euclidean')
        max_distance = np.max(distance_matrix)
        similarity_matrix = 1 - (distance_matrix / max_distance)
    else:
        raise ValueError("similarity_metric must be 'cosine', 'correlation', or 'euclidean'")
    
    # Create NetworkX graph
    G = nx.Graph()
    n_nodes = data.shape[0]
    
    # Add nodes
    for i in range(n_nodes):
        G.add_node(i)
    
    # Add edges based on similarity threshold
    edges_added = 0
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            similarity = similarity_matrix[i, j]
            if similarity >= similarity_threshold:
                G.add_edge(i, j, weight=similarity)
                edges_added += 1
    
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    print(f"Edges added: {edges_added} (threshold: {similarity_threshold})")
    
    # Check connectivity
    if nx.is_connected(G):
        print("Graph is connected")
    else:
        components = list(nx.connected_components(G))
        print(f"Graph has {len(components)} connected components")
        print(f"Largest component size: {len(max(components, key=len))}")
    
    return G, similarity_matrix

# NEW: Function to optimize Louvain clustering parameters
def optimize_louvain_parameters(data, output_dir, test_resolutions=True):
    """
    Test different similarity thresholds, metrics, and resolutions to find optimal Louvain clustering.
    """
    print("Optimizing Louvain clustering parameters...")
    
    # Test different parameters
    similarity_metrics = ['cosine', 'correlation']
    threshold_ranges = {
        'cosine': np.arange(0.1, 0.8, 0.05),
        'correlation': np.arange(0.2, 0.9, 0.05)
    }
    
    # Test different resolution values if requested
    if test_resolutions:
        resolution_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]
        print(f"Testing {len(resolution_values)} resolution values: {resolution_values}")
    else:
        resolution_values = [1.0]  # Default only
    
    results = []
    
    for metric in similarity_metrics:
        print(f"\nTesting {metric} similarity...")
        thresholds = threshold_ranges[metric]
        
        for threshold in thresholds:
            for resolution in resolution_values:
                try:
                    # Create graph
                    G, _ = create_similarity_graph(data, 
                                                 similarity_threshold=threshold, 
                                                 similarity_metric=metric)
                    
                    # Skip if graph is too sparse or dense
                    if G.number_of_edges() < 10 or G.number_of_edges() > data.shape[0] * (data.shape[0] - 1) * 0.3:
                        continue
                    
                    # Apply Louvain clustering
                    if LOUVAIN_AVAILABLE:
                        partition = community_louvain.best_partition(G, resolution=resolution, random_state=42)
                        modularity = community_louvain.modularity(partition, G)
                    else:
                        # Use NetworkX fallback
                        try:
                            communities = nx_community.louvain_communities(G, resolution=resolution, seed=42)
                            partition = {}
                            for i, community in enumerate(communities):
                                for node in community:
                                    partition[node] = i
                            modularity = nx_community.modularity(G, communities)
                        except:
                            # Final fallback
                            communities = nx_community.greedy_modularity_communities(G)
                            partition = {}
                            for i, community in enumerate(communities):
                                for node in community:
                                    partition[node] = i
                            modularity = nx_community.modularity(G, communities)
                    
                    # Calculate metrics
                    n_communities = len(set(partition.values()))
                    
                    # Check if communities are well-distributed
                    community_sizes = Counter(partition.values())
                    size_std = np.std(list(community_sizes.values()))
                    min_community_size = min(community_sizes.values())
                    max_community_size = max(community_sizes.values())
                    
                    results.append({
                        'metric': metric,
                        'threshold': threshold,
                        'resolution': resolution,
                        'n_communities': n_communities,
                        'modularity': modularity,
                        'edges': G.number_of_edges(),
                        'size_std': size_std,
                        'min_size': min_community_size,
                        'max_size': max_community_size,
                        'size_ratio': max_community_size / min_community_size if min_community_size > 0 else np.inf
                    })
                    
                    if test_resolutions:
                        print(f"  {metric} T:{threshold:.2f} R:{resolution:.1f}: {n_communities} communities, mod: {modularity:.3f}")
                    else:
                        print(f"  Threshold {threshold:.2f}: {n_communities} communities, modularity: {modularity:.3f}")
                        
                except Exception as e:
                    print(f"  Error with {metric} T:{threshold:.2f} R:{resolution:.1f}: {e}")
                    continue
    
    # Convert to DataFrame and analyze
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("No valid parameter combinations found!")
        return 'cosine', 0.3, 1.0
    
    # Save results
    results_file = os.path.join(output_dir, "louvain_parameter_optimization_with_resolution.csv")
    results_df.to_csv(results_file, index=False)
    
    # Find best parameters based on modularity and community distribution
    # Penalize extreme size ratios and very few/many communities
    results_df['score'] = (
        results_df['modularity'] * 0.6 +  # Primary criterion
        (1 / (1 + results_df['size_ratio'])) * 0.2 +  # Penalize unbalanced communities
        (1 / (1 + abs(results_df['n_communities'] - 10))) * 0.2  # Prefer moderate number of communities
    )
    
    # Filter out extreme cases
    filtered_results = results_df[
        (results_df['n_communities'] >= 3) & 
        (results_df['n_communities'] <= 50) &
        (results_df['modularity'] > 0.1) &
        (results_df['size_ratio'] < 10)
    ]
    
    if len(filtered_results) == 0:
        print("No suitable parameter combinations found after filtering!")
        best_params = results_df.loc[results_df['modularity'].idxmax()]
    else:
        best_params = filtered_results.loc[filtered_results['score'].idxmax()]
    
    print(f"\nBest parameters:")
    print(f"  Similarity metric: {best_params['metric']}")
    print(f"  Threshold: {best_params['threshold']:.3f}")
    print(f"  Resolution: {best_params['resolution']:.1f}")
    print(f"  Number of communities: {best_params['n_communities']}")
    print(f"  Modularity: {best_params['modularity']:.3f}")
    print(f"  Size ratio: {best_params['size_ratio']:.2f}")
    
    print(f"Parameter optimization results saved to: {results_file}")
    
    return best_params['metric'], best_params['threshold'], best_params['resolution']

# NEW: Function to apply Louvain clustering with fallback options and resolution control
def apply_louvain_clustering(data, similarity_metric='cosine', similarity_threshold=0.3, 
                           resolution=1.0, output_dir=None):
    """
    Apply Louvain clustering to find global CNs with fallback to NetworkX if python-louvain unavailable.
    
    Parameters:
    - resolution: Controls community detection granularity. Higher = more communities.
                 Default 1.0. Try 1.5-3.0 for finer clustering.
    
    Returns:
    - global_cn_labels: Array of global CN assignments
    - modularity: Modularity score of the clustering
    - graph_info: Dictionary with graph statistics
    """
    print(f"Applying community detection clustering...")
    print(f"Similarity metric: {similarity_metric}")
    print(f"Similarity threshold: {similarity_threshold}")
    print(f"Resolution parameter: {resolution}")
    
    # Create similarity graph
    G, similarity_matrix = create_similarity_graph(data, 
                                                  similarity_threshold=similarity_threshold,
                                                  similarity_metric=similarity_metric)
    
    # Apply community detection
    print("Running community detection...")
    
    if LOUVAIN_AVAILABLE:
        # Use python-louvain package (preferred)
        partition = community_louvain.best_partition(G, resolution=resolution, random_state=42)
        modularity = community_louvain.modularity(partition, G)
        method_used = f"Louvain (python-louvain, resolution={resolution})"
    else:
        # Fallback to NetworkX community detection
        try:
            # Use Louvain method from NetworkX (requires networkx >= 2.5)
            communities = nx_community.louvain_communities(G, resolution=resolution, seed=42)
            # Convert to partition dictionary format
            partition = {}
            for i, community in enumerate(communities):
                for node in community:
                    partition[node] = i
            modularity = nx_community.modularity(G, communities)
            method_used = f"Louvain (NetworkX, resolution={resolution})"
        except (AttributeError, TypeError):
            # Fallback to greedy modularity communities (doesn't support resolution)
            print(f"Louvain with resolution not available, using greedy modularity clustering...")
            print(f"Note: Greedy modularity doesn't support resolution parameter")
            communities = nx_community.greedy_modularity_communities(G)
            partition = {}
            for i, community in enumerate(communities):
                for node in community:
                    partition[node] = i
            modularity = nx_community.modularity(G, communities)
            method_used = "Greedy Modularity (NetworkX, no resolution control)"
    
    print(f"Community detection method used: {method_used}")
    
    # Convert partition to labels array
    global_cn_labels = np.array([partition.get(i, -1) for i in range(data.shape[0])])
    
    # Get community statistics
    unique_communities = set(global_cn_labels)
    n_communities = len(unique_communities)
    community_sizes = Counter(global_cn_labels)
    
    # Calculate graph statistics
    graph_info = {
        'n_nodes': G.number_of_nodes(),
        'n_edges': G.number_of_edges(),
        'density': nx.density(G),
        'n_communities': n_communities,
        'modularity': modularity,
        'resolution': resolution,
        'community_sizes': dict(community_sizes),
        'avg_community_size': np.mean(list(community_sizes.values())),
        'std_community_size': np.std(list(community_sizes.values())),
        'method_used': method_used
    }
    
    print(f"Community detection completed:")
    print(f"  Number of communities: {n_communities}")
    print(f"  Modularity: {modularity:.3f}")
    print(f"  Average community size: {graph_info['avg_community_size']:.1f}")
    print(f"  Community size std: {graph_info['std_community_size']:.1f}")
    
    return global_cn_labels, modularity, graph_info

# NEW: Function to create leave-one-out z-score heatmaps - UPDATED WITH BORDERS
def create_leave_one_out_zscore_heatmap(global_cn_data, output_dir):
    """
    Create leave-one-out z-score heatmaps for each global CN to identify characteristic cell types.
    Cell types ordered by CELLCAT3_ORDER, z-scores clamped to [-10, 10], with average cell counts.
    """
    print("Creating leave-one-out z-score heatmaps...")
    
    # Get all cell type columns (excluding metadata columns)
    feature_cols = [col for col in global_cn_data.columns 
                   if col not in ['CN', 'CompName', 'TotalCells', 'Dataset', 'GlobalCN']]
    
    # Order cell types according to CELLCAT3_ORDER
    ordered_feature_cols = []
    for cell_type in CELLCAT3_ORDER:
        if cell_type in feature_cols:
            ordered_feature_cols.append(cell_type)
    
    # Add any remaining cell types that weren't in CELLCAT3_ORDER
    remaining_cols = [col for col in feature_cols if col not in ordered_feature_cols]
    ordered_feature_cols.extend(sorted(remaining_cols))
    
    print(f"Ordered {len(ordered_feature_cols)} cell types according to CELLCAT3_ORDER")
    
    # Calculate global CN profiles
    global_cn_profiles = global_cn_data.groupby('GlobalCN')[ordered_feature_cols].mean()
    global_cn_cell_counts = global_cn_data.groupby('GlobalCN')['TotalCells'].mean().astype(int)
    
    # Calculate leave-one-out z-scores
    loo_zscores = pd.DataFrame(index=global_cn_profiles.index, columns=ordered_feature_cols)
    
    for global_cn in global_cn_profiles.index:
        for cell_type in ordered_feature_cols:
            # Calculate mean without this global CN (leave-one-out)
            other_global_cns = global_cn_data[global_cn_data['GlobalCN'] != global_cn]
            
            if len(other_global_cns) > 0:
                # Mean and std of other global CNs
                loo_mean = other_global_cns[cell_type].mean()
                loo_std = other_global_cns[cell_type].std()
                
                # Current global CN mean
                current_mean = global_cn_profiles.loc[global_cn, cell_type]
                
                # Calculate z-score
                if loo_std > 0:
                    zscore = (current_mean - loo_mean) / loo_std
                else:
                    zscore = 0
                
                # Clamp z-score to [-2.5, 2.5] range
                zscore = np.clip(zscore, -2.5, 2.5)
                
                loo_zscores.loc[global_cn, cell_type] = zscore
            else:
                loo_zscores.loc[global_cn, cell_type] = 0
    
    # Convert to numeric
    loo_zscores = loo_zscores.astype(float)
    
    # Add average cell counts as the rightmost column
    loo_zscores['Avg Total Cells'] = global_cn_cell_counts
    
    # Create heatmap
    plt.figure(figsize=(24, 12))
    
    # Create custom y-tick labels (just Global CN numbers for cleaner look)
    ytick_labels = [f"{int(cn)}" for cn in loo_zscores.index]
    
    # Create custom colormap for the heatmap
    # Cell counts column should have different coloring
    plot_data = loo_zscores.copy()
    
    # Normalize cell counts to z-score range for visualization
    cell_counts = plot_data['Avg Total Cells'].values
    if len(cell_counts) > 1:
        # Scale cell counts to roughly [-2, 2] range for better visualization
        cell_counts_normalized = 4 * (cell_counts - cell_counts.min()) / (cell_counts.max() - cell_counts.min()) - 2
        plot_data['Avg Total Cells'] = cell_counts_normalized
    
    # Create heatmap with diverging colormap centered at 0
    ax = sns.heatmap(plot_data, 
                     cmap='RdBu_r', 
                     center=0,
                     vmin=-2.5,
                     vmax=2.5,
                     annot=False,
                     fmt='',
                     linewidths=0.5,    # UPDATED: Thin black borders
                     linecolor='black', # UPDATED: Explicit black color
                     square=True,       # UPDATED: Square boxes
                     cbar_kws={'label': 'Leave-One-Out Z-Score', 'shrink': 0.8})
    
    # Customize y-axis
    ax.set_yticklabels(ytick_labels, rotation=0, fontsize=12)
    ax.set_ylabel('Global CN (Louvain Communities)', fontsize=16, fontweight='bold')
    
    # Customize x-axis
    xlabels = list(ordered_feature_cols) + ['Avg Total Cells']
    ax.set_xticklabels(xlabels, rotation=90, ha='center', fontsize=12)
    ax.set_xlabel('Cell Types (Ordered by Category)', fontsize=16, fontweight='bold')
    
    # Add vertical line to separate cell types from cell count column
    ax.axvline(x=len(ordered_feature_cols), color='black', linewidth=2)
    
    plt.title('Leave-One-Out Z-Scores for Cell Types in Global CNs (Louvain)\n(Range: -2.5 to +2.5, Positive = Enriched, Negative = Depleted)', 
              fontsize=16, pad=40, fontweight='bold')
    
    plt.tight_layout()
    
    # Save heatmap
    loo_file = os.path.join(output_dir, "global_CN_leave_one_out_zscores_louvain.png")
    plt.savefig(loo_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    # Save z-score data (without the normalized cell counts column)
    zscore_csv = os.path.join(output_dir, "global_CN_leave_one_out_zscores_louvain.csv")
    loo_zscores_original = loo_zscores.copy()
    loo_zscores_original['Avg Total Cells'] = global_cn_cell_counts  # Restore original cell counts
    loo_zscores_original.to_csv(zscore_csv)
    
    print(f"Leave-one-out z-score heatmap (Louvain) saved to: {loo_file}")
    print(f"Z-score data (Louvain) saved to: {zscore_csv}")
    
    return loo_zscores_original

def create_complete_patient_density_matrix(patient_level_data, data_path):
    """
    Create a complete patient-by-global-CN density matrix with zeros for missing combinations.
    Ensures all patients from each dataset are included for every global CN.
    """
    print("🔄 Creating complete patient-density matrix with zeros for missing combinations...")
    
    # Load the original data to get complete patient roster
    dfInfo = pd.read_csv(data_path, low_memory=False)
    
    # Get complete patient roster for each dataset
    complete_patient_roster = dfInfo.groupby(['Dataset', 'Patient']).size().reset_index(name='CellCount')
    complete_patient_roster = complete_patient_roster[['Dataset', 'Patient']]
    
    print(f"📊 Complete patient roster:")
    for dataset in complete_patient_roster['Dataset'].unique():
        n_patients = len(complete_patient_roster[complete_patient_roster['Dataset'] == dataset])
        patients = complete_patient_roster[complete_patient_roster['Dataset'] == dataset]['Patient'].unique()
        print(f"  {dataset}: {n_patients} patients - {list(patients)}")
    
    # Get all unique global CNs
    all_global_cns = sorted(patient_level_data['GlobalCN'].unique())
    print(f"📊 Global CNs to analyze: {all_global_cns}")
    
    # Create complete combinations of Dataset-Patient-GlobalCN
    complete_combinations = []
    for _, patient_row in complete_patient_roster.iterrows():
        for global_cn in all_global_cns:
            complete_combinations.append({
                'Dataset': patient_row['Dataset'],
                'Patient': patient_row['Patient'],
                'GlobalCN': global_cn
            })
    
    complete_df = pd.DataFrame(complete_combinations)
    print(f"📊 Complete combinations created: {len(complete_df)} records")
    
    # Merge with actual data, filling missing with zeros
    complete_matrix = complete_df.merge(
        patient_level_data[['Dataset', 'Patient', 'GlobalCN', 'LocalCN_Count', 'TissueArea_mm2', 'Density_LocalCNs_per_mm2']], 
        on=['Dataset', 'Patient', 'GlobalCN'], 
        how='left'
    )
    
    # Fill missing values with zeros
    complete_matrix['LocalCN_Count'] = complete_matrix['LocalCN_Count'].fillna(0)
    complete_matrix['TissueArea_mm2'] = complete_matrix['TissueArea_mm2'].fillna(0)
    complete_matrix['Density_LocalCNs_per_mm2'] = complete_matrix['Density_LocalCNs_per_mm2'].fillna(0)
    
    # For patients with zero local CNs, we need to get their total tissue area from the original data
    # This ensures density calculation is meaningful (0 local CNs / actual tissue area = 0 density)
    for idx, row in complete_matrix.iterrows():
        if row['TissueArea_mm2'] == 0:  # Missing tissue area
            # Get tissue area for this patient from any global CN they do have
            patient_tissue_data = patient_level_data[
                (patient_level_data['Dataset'] == row['Dataset']) & 
                (patient_level_data['Patient'] == row['Patient'])
            ]
            if len(patient_tissue_data) > 0:
                # Use the tissue area from this patient's other global CNs
                avg_tissue_area = patient_tissue_data['TissueArea_mm2'].mean()
                complete_matrix.loc[idx, 'TissueArea_mm2'] = avg_tissue_area
                # Density remains 0 (0 local CNs / tissue area = 0)
                complete_matrix.loc[idx, 'Density_LocalCNs_per_mm2'] = 0
    
    print(f"📊 Complete matrix summary:")
    print(f"  Total records: {len(complete_matrix)}")
    print(f"  Records with zero density: {len(complete_matrix[complete_matrix['Density_LocalCNs_per_mm2'] == 0])}")
    print(f"  Records with non-zero density: {len(complete_matrix[complete_matrix['Density_LocalCNs_per_mm2'] > 0])}")
    
    # Verify patient counts per dataset
    for dataset in complete_matrix['Dataset'].unique():
        n_patients = len(complete_matrix[complete_matrix['Dataset'] == dataset]['Patient'].unique())
        print(f"  {dataset}: {n_patients} patients per global CN")
    
    return complete_matrix

# UPDATED: Modified to use tissue masks instead of inflammation masks
def analyze_global_cn_density_patient_level(global_cn_data, data_path, datasets, output_dir, epsilon, min_samples, filter_structural=True):
    """
    PATIENT-LEVEL ANALYSIS: Enhanced analysis with complete patient inclusion (zeros for missing)
    Uses TISSUE MASKS instead of inflammation masks for area calculation.
    """
    print("Analyzing local CN density per global CN by PATIENT (Patient-level Analysis with tissue mask areas)...")
    
    # CRITICAL FIX: Increase PIL image size limit for large tissue masks
    Image.MAX_IMAGE_PIXELS = None  # Remove size limit completely
    
    # Load the main data to get patient information
    dfInfo = pd.read_csv(data_path, low_memory=False)
    
    # Debug: Check dataset names in the data
    print(f"Datasets found in CSV: {dfInfo['Dataset'].unique()}")
    
    # Map dataset names if they're different in the CSV vs. what we expect
    dataset_mapping = {
        'IgG4-RKD': 'IgG4',
        'KC': 'NK2',
        'AAV': 'AAV'
    }
    
    # Apply reverse mapping to use actual dataset names from CSV
    actual_datasets = []
    for dataset in datasets:
        # Find the actual dataset name in the CSV
        actual_name = dataset
        for csv_name, expected_name in dataset_mapping.items():
            if expected_name == dataset and csv_name in dfInfo['Dataset'].unique():
                actual_name = csv_name
                break
        actual_datasets.append(actual_name)
    
    print(f"Mapped datasets: {dict(zip(datasets, actual_datasets))}")
    
    # Create density analysis results - PATIENT LEVEL
    successful_compartments = 0
    total_compartments = 0
    
    # Base path for tissue masks
    base_path = '/project/mclark/SCAMPI_datasets'
    
    # First collect all compartment-level data, then aggregate by patient
    compartment_results = []
    
    # Process each dataset using actual names
    for dataset_idx, dataset in enumerate(datasets):
        actual_dataset = actual_datasets[dataset_idx]
        print(f"  Processing tissue mask density analysis for dataset: {dataset} (CSV name: {actual_dataset})")
        
        # Filter data for current dataset using actual name
        dataset_data = dfInfo[dfInfo['Dataset'] == actual_dataset]
        
        if len(dataset_data) == 0:
            print(f"    No data found for dataset {actual_dataset}, skipping...")
            continue
        
        # Get unique compartments and patients
        compartments = dataset_data['CompName'].unique()
        patients = dataset_data['Patient'].unique() if 'Patient' in dataset_data.columns else [f'{actual_dataset}_Unknown']
        
        print(f"    Found {len(compartments)} compartments and {len(patients)} patients")
        
        for comp in compartments:
            total_compartments += 1
            print(f"    Processing compartment: {comp}")
            
            # Get compartment data
            comp_data = dataset_data[dataset_data['CompName'] == comp]
            
            # Use the expected dataset name for file paths (not the CSV name)
            dataset_name_for_path = dataset
            comp_name = comp
            
            # UPDATED: Define paths to the tissue mask files
            tissue_mask_path1 = f'{base_path}/{dataset_name_for_path}/tissue_composite_masks/DAPI_UV_1_{dataset_name_for_path}_{comp_name}.tif'
            tissue_mask_path2 = f'{base_path}/{dataset_name_for_path}/tissue_composite_masks/DAPI_UV_2_{dataset_name_for_path}_{comp_name}.tif'
            tissue_mask_path1_alt = f'{base_path}/{dataset_name_for_path}/tissue_composite_masks/DAPI_UV_1_{dataset_name_for_path}_{comp_name}.tiff'
            tissue_mask_path2_alt = f'{base_path}/{dataset_name_for_path}/tissue_composite_masks/DAPI_UV_2_{dataset_name_for_path}_{comp_name}.tiff'
            
            # Check which tissue mask path exists
            tissue_mask_path = None
            for path in [tissue_mask_path1, tissue_mask_path2, tissue_mask_path1_alt, tissue_mask_path2_alt]:
                if os.path.exists(path):
                    tissue_mask_path = path
                    break
            
            if not tissue_mask_path:
                print(f"      No tissue mask found for {comp_name}, skipping...")
                continue
            
            try:
                # Load tissue mask with better error handling and memory management
                print(f"      Loading tissue mask: {os.path.basename(tissue_mask_path)}")
                
                # Get file size for monitoring
                file_size_mb = os.path.getsize(tissue_mask_path) / (1024 * 1024)
                print(f"      File size: {file_size_mb:.1f} MB")
                
                # Load image with explicit mode handling
                with Image.open(tissue_mask_path) as tissue_mask:
                    print(f"      Image size: {tissue_mask.size} pixels ({tissue_mask.mode} mode)")
                    
                    # UPDATED: Convert to numpy array and check for any pixel value > 0 (tissue area)
                    if tissue_mask.mode in ['L', '1']:  # Grayscale or binary
                        tissue_array = np.array(tissue_mask) > 0
                    elif tissue_mask.mode in ['RGB', 'RGBA']:  # Color images
                        # Convert to grayscale first, then threshold for > 0
                        tissue_gray = tissue_mask.convert('L')
                        tissue_array = np.array(tissue_gray) > 0
                    else:
                        print(f"      Unexpected image mode: {tissue_mask.mode}, converting to grayscale")
                        tissue_gray = tissue_mask.convert('L')
                        tissue_array = np.array(tissue_gray) > 0
                
                tissue_area_pixels = np.sum(tissue_array)
                print(f"      Tissue pixels: {tissue_area_pixels:,}")
                
                # UPDATED: Convert to area in mm² using correct pixel size
                pixels_per_mm2 = (1000 / 0.1507) ** 2  # 1 pixel = 0.1507 µm
                tissue_area_mm2 = tissue_area_pixels / pixels_per_mm2
                
                print(f"      Tissue area: {tissue_area_mm2:.2f} mm²")
                
                # Check if tissue area is zero
                if tissue_area_mm2 == 0:
                    print(f"      No tissue area found in {comp_name}, skipping...")
                    continue
                
                # Load DBSCAN results for this compartment
                suffix = "_immune_only" if filter_structural else ""
                dbscan_dir = f'/project/mclark/SCAMPI_datasets/{dataset_name_for_path}/DBSCAN/DBSCAN_ep_{epsilon}_ms_{min_samples}{suffix}'
                dbscan_file = f'DBSCAN_{comp}_ms_{min_samples}_ep_{epsilon}.csv'
                dbscan_path = os.path.join(dbscan_dir, dbscan_file)
                
                if not os.path.exists(dbscan_path):
                    print(f"      No DBSCAN results found for {comp}, skipping...")
                    continue
                
                # Read DBSCAN results
                dbscan_result = pd.read_csv(dbscan_path)
                
                # Merge with compartment data
                comp_data_with_cn = comp_data.merge(dbscan_result[['DBSCAN_label', 'CellID']], 
                                                   on='CellID', how='left')
                
                # Get global CN mapping from global_cn_data
                comp_global_mapping = {}
                comp_global_data = global_cn_data[
                    (global_cn_data['Dataset'] == actual_dataset) & 
                    (global_cn_data['CompName'] == comp)
                ]
                
                for _, row in comp_global_data.iterrows():
                    local_cn = row['CN']
                    global_cn = row['GlobalCN']
                    comp_global_mapping[local_cn] = global_cn
                
                # Map local CNs to global CNs
                comp_data_with_cn['GlobalCN'] = comp_data_with_cn['DBSCAN_label'].map(comp_global_mapping)
                comp_data_with_cn['GlobalCN'] = comp_data_with_cn['GlobalCN'].fillna(-1)
                
                # Store compartment-level results for later patient aggregation
                if 'Patient' in comp_data_with_cn.columns:
                    for patient in comp_data_with_cn['Patient'].unique():
                        patient_data = comp_data_with_cn[comp_data_with_cn['Patient'] == patient]
                        
                        # Count local CNs per global CN for this patient in this compartment
                        local_cn_counts = patient_data.groupby(['GlobalCN', 'DBSCAN_label']).size().reset_index(name='CellCount')
                        global_cn_counts = local_cn_counts.groupby('GlobalCN').size().reset_index(name='LocalCN_Count')
                        
                        # Store compartment-level data for patient aggregation
                        for _, row in global_cn_counts.iterrows():
                            global_cn = row['GlobalCN']
                            local_cn_count = row['LocalCN_Count']
                            
                            if global_cn != -1:  # Exclude noise
                                compartment_results.append({
                                    'Dataset': dataset,  # Use expected dataset name for consistency
                                    'Dataset_CSV_Name': actual_dataset,  # Store actual CSV name for reference
                                    'Compartment': comp,
                                    'Patient': patient,
                                    'GlobalCN': global_cn,
                                    'LocalCN_Count': local_cn_count,
                                    'TissueArea_mm2': tissue_area_mm2,  # UPDATED: Use tissue area
                                    'Density_LocalCNs_per_mm2': local_cn_count / tissue_area_mm2 if tissue_area_mm2 > 0 else 0
                                })
                else:
                    # If no patient column, treat as single patient per dataset
                    local_cn_counts = comp_data_with_cn.groupby(['GlobalCN', 'DBSCAN_label']).size().reset_index(name='CellCount')
                    global_cn_counts = local_cn_counts.groupby('GlobalCN').size().reset_index(name='LocalCN_Count')
                    
                    for _, row in global_cn_counts.iterrows():
                        global_cn = row['GlobalCN']
                        local_cn_count = row['LocalCN_Count']
                        
                        if global_cn != -1:  # Exclude noise
                            compartment_results.append({
                                'Dataset': dataset,
                                'Dataset_CSV_Name': actual_dataset,
                                'Compartment': comp,
                                'Patient': f'{dataset}_Patient1',
                                'GlobalCN': global_cn,
                                'LocalCN_Count': local_cn_count,
                                'TissueArea_mm2': tissue_area_mm2,  # UPDATED: Use tissue area
                                'Density_LocalCNs_per_mm2': local_cn_count / tissue_area_mm2 if tissue_area_mm2 > 0 else 0
                            })
                
                successful_compartments += 1
                print(f"      ✅ Successfully processed {comp}")
                
                # Clean up memory
                del tissue_array
                
            except Exception as e:
                print(f"      ❌ Error processing {comp}: {e}")
                continue
    
    # PATIENT-LEVEL AGGREGATION
    print(f"\n📊 AGGREGATING TO PATIENT LEVEL...")
    compartment_df = pd.DataFrame(compartment_results)
    
    if len(compartment_df) == 0:
        print("❌ No compartment data to aggregate.")
        return None, None, None
    
    # Aggregate to patient level: sum local CN counts and tissue areas across compartments for each patient
    print(f"  Compartment-level records: {len(compartment_df)}")
    print(f"  Unique patients: {len(compartment_df['Patient'].unique())}")
    print(f"  Unique global CNs: {len(compartment_df['GlobalCN'].unique())}")
    
    # UPDATED: Group by Dataset, Patient, GlobalCN and sum the counts and tissue areas
    patient_level_data = compartment_df.groupby(['Dataset', 'Dataset_CSV_Name', 'Patient', 'GlobalCN']).agg({
        'LocalCN_Count': 'sum',  # Sum local CN counts across compartments
        'TissueArea_mm2': 'sum'   # Sum tissue areas across compartments
    }).reset_index()
    
    # Recalculate density at patient level
    patient_level_data['Density_LocalCNs_per_mm2'] = (
        patient_level_data['LocalCN_Count'] / patient_level_data['TissueArea_mm2']
    ).fillna(0)
    
    print(f"  Patient-level records (before completing matrix): {len(patient_level_data)}")
    
    # NEW: Create complete patient-density matrix with zeros for missing combinations
    complete_patient_data = create_complete_patient_density_matrix(patient_level_data, data_path)
    
    # Print processing summary
    print(f"\n📊 PROCESSING SUMMARY:")
    print(f"  Total compartments attempted: {total_compartments}")
    print(f"  Successfully processed: {successful_compartments}")
    print(f"  Success rate: {successful_compartments/total_compartments*100:.1f}%")
    print(f"  Final patient-level records (complete): {len(complete_patient_data)}")
    
    # Print dataset breakdown
    for dataset in complete_patient_data['Dataset'].unique():
        dataset_records = len(complete_patient_data[complete_patient_data['Dataset'] == dataset])
        dataset_patients = len(complete_patient_data[complete_patient_data['Dataset'] == dataset]['Patient'].unique())
        print(f"  {dataset}: {dataset_records} records from {dataset_patients} patients")
    
    # UPDATED: Save patient-level density data with tissue area
    density_csv = os.path.join(output_dir, "global_CN_density_by_patient_PATIENT_LEVEL_TISSUE_MASK.csv")
    complete_patient_data.to_csv(density_csv, index=False)
    print(f"💾 Complete patient-level density data (tissue mask) saved to: {density_csv}")
    
    # Perform statistical analysis at patient level with complete data
    stats_results = perform_statistical_analysis_patient_level(complete_patient_data, output_dir)
    
    # Perform per-global-CN statistical analysis with complete data
    per_gcn_stats_results = perform_statistical_analysis_per_global_cn(complete_patient_data, output_dir)
    
    return complete_patient_data, stats_results, per_gcn_stats_results

def perform_statistical_analysis_patient_level(density_df, output_dir):
    """
    Enhanced statistical analysis at PATIENT level
    """
    print("🧮 Performing statistical analysis of local CN densities at PATIENT LEVEL...")
    
    # The density_df should already be at patient level
    print(f"📊 Patient-level data: {len(density_df)} records")
    print(f"📊 Unique patients: {len(density_df['Patient'].unique())}")
    print(f"📊 Datasets: {density_df['Dataset'].unique()}")
    
    # Calculate total density per patient across all global CNs
    patient_total_densities = density_df.groupby(['Dataset', 'Patient'])['Density_LocalCNs_per_mm2'].sum().reset_index()
    patient_total_densities['GlobalCN'] = 'Total'
    patient_total_densities = patient_total_densities[['Dataset', 'Patient', 'GlobalCN', 'Density_LocalCNs_per_mm2']]
    
    # Combine specific and total densities
    all_patient_data = pd.concat([density_df[['Dataset', 'Patient', 'GlobalCN', 'Density_LocalCNs_per_mm2']], 
                                 patient_total_densities], ignore_index=True)
    
    print(f"📊 Total patient records for analysis: {len(patient_total_densities)}")
    
    # Print patient breakdown by dataset
    for dataset in patient_total_densities['Dataset'].unique():
        n_patients = len(patient_total_densities[patient_total_densities['Dataset'] == dataset])
        print(f"  {dataset}: {n_patients} patients")
    
    # Statistical tests
    stats_results = {}
    
    # 1. Overall comparison across datasets (total density)
    total_density_by_dataset = {}
    datasets = patient_total_densities['Dataset'].unique()
    
    for dataset in datasets:
        total_density_by_dataset[dataset] = patient_total_densities[
            patient_total_densities['Dataset'] == dataset
        ]['Density_LocalCNs_per_mm2'].values
    
    print(f"📈 Dataset sample sizes: {[(d, len(total_density_by_dataset[d])) for d in datasets]}")
    
    if len(datasets) >= 2:
        # Kruskal-Wallis test for overall differences
        if len(datasets) > 2:
            h_stat, p_kruskal = kruskal(*[total_density_by_dataset[d] for d in datasets])
            stats_results['overall_kruskal'] = {
                'statistic': h_stat,
                'p_value': p_kruskal,
                'datasets': list(datasets),
                'sample_sizes': [len(total_density_by_dataset[d]) for d in datasets]
            }
            print(f"🎯 Kruskal-Wallis test (overall): H={h_stat:.3f}, p={p_kruskal:.6f}")
        
        # Pairwise comparisons with multiple testing correction
        pairwise_results = []
        dataset_pairs = list(combinations(datasets, 2))
        
        for dataset1, dataset2 in dataset_pairs:
            data1 = total_density_by_dataset[dataset1]
            data2 = total_density_by_dataset[dataset2]
            
            if len(data1) > 0 and len(data2) > 0:
                # Mann-Whitney U test
                u_stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                
                pairwise_results.append({
                    'dataset1': dataset1,
                    'dataset2': dataset2,
                    'u_statistic': u_stat,
                    'p_value': p_value,
                    'n1': len(data1),
                    'n2': len(data2),
                    'median1': np.median(data1),
                    'median2': np.median(data2),
                    'mean1': np.mean(data1),
                    'mean2': np.mean(data2)
                })
        
        # Multiple testing correction
        if pairwise_results:
            p_values = [result['p_value'] for result in pairwise_results]
            corrected_p = multipletests(p_values, method='fdr_bh')[1]  # Benjamini-Hochberg
            
            for i, result in enumerate(pairwise_results):
                result['p_corrected'] = corrected_p[i]
                result['significant'] = corrected_p[i] < 0.05
            
            stats_results['pairwise_total_density'] = pairwise_results
        
        print(f"✅ Completed {len(pairwise_results)} pairwise comparisons at patient level")
    else:
        print("⚠️  Only one dataset available - no statistical comparisons possible")
    
    # Save statistical results
    save_statistical_results_patient_level(stats_results, output_dir)
    
    return stats_results

def save_statistical_results_patient_level(stats_results, output_dir):
    """
    Save statistical results for patient-level analysis
    """
    # Save overall pairwise comparisons
    if 'pairwise_total_density' in stats_results:
        pairwise_df = pd.DataFrame(stats_results['pairwise_total_density'])
        pairwise_file = os.path.join(output_dir, "statistical_analysis_pairwise_total_density_PATIENT_LEVEL_TISSUE.csv")
        pairwise_df.to_csv(pairwise_file, index=False)
        print(f"📊 Patient-level pairwise comparison results (tissue mask) saved to: {pairwise_file}")
    
    # Save summary statistics
    summary_file = os.path.join(output_dir, "statistical_analysis_summary_PATIENT_LEVEL_TISSUE.txt")
    with open(summary_file, 'w') as f:
        f.write("STATISTICAL ANALYSIS SUMMARY (PATIENT LEVEL - TISSUE MASK)\n")
        f.write("="*70 + "\n\n")
        
        # Overall Kruskal-Wallis
        if 'overall_kruskal' in stats_results:
            result = stats_results['overall_kruskal']
            f.write("OVERALL COMPARISON (Total Density - Patient Level - Tissue Area)\n")
            f.write("-"*50 + "\n")
            f.write(f"Kruskal-Wallis H statistic: {result['statistic']:.3f}\n")
            f.write(f"P-value: {result['p_value']:.6f}\n")
            f.write(f"Datasets: {result['datasets']}\n")
            f.write(f"Sample sizes (patients): {result['sample_sizes']}\n")
            if result['p_value'] < 0.05:
                f.write("Result: SIGNIFICANT difference between datasets at patient level\n")
            else:
                f.write("Result: No significant difference between datasets at patient level\n")
            f.write("\n")
        
        # Pairwise comparisons
        if 'pairwise_total_density' in stats_results:
            f.write("PAIRWISE COMPARISONS (Total Density - Patient Level - Tissue Area)\n")
            f.write("-"*55 + "\n")
            for result in stats_results['pairwise_total_density']:
                f.write(f"{result['dataset1']} vs {result['dataset2']}:\n")
                f.write(f"  Mann-Whitney U: {result['u_statistic']:.3f}\n")
                f.write(f"  Raw p-value: {result['p_value']:.6f}\n")
                f.write(f"  Corrected p-value (FDR): {result['p_corrected']:.6f}\n")
                f.write(f"  Medians: {result['median1']:.4f} vs {result['median2']:.4f}\n")
                f.write(f"  Sample sizes (patients): {result['n1']} vs {result['n2']}\n")
                f.write(f"  Significant: {'YES' if result['significant'] else 'NO'}\n")
                f.write("\n")
        else:
            f.write("PAIRWISE COMPARISONS: Not possible (insufficient datasets)\n\n")
    
    print(f"📄 Patient-level statistical summary (tissue mask) saved to: {summary_file}")

def perform_statistical_analysis_per_global_cn(density_df, output_dir):
    """
    Perform statistical comparisons across datasets for each global CN separately.
    """
    print("🧮 Performing statistical analysis per global CN...")
    
    # Prepare data - group by patient and dataset to get patient-level densities
    patient_densities = density_df.groupby(['Dataset', 'Patient', 'GlobalCN'])['Density_LocalCNs_per_mm2'].mean().reset_index()
    
    # Get unique datasets and global CNs
    datasets = sorted(patient_densities['Dataset'].unique())
    global_cns = sorted(patient_densities['GlobalCN'].unique())
    
    print(f"📊 Analyzing {len(global_cns)} global CNs across {len(datasets)} datasets")
    print(f"📊 Datasets: {datasets}")
    
    # Store results for each global CN
    all_results = []
    
    for global_cn in global_cns:
        print(f"  Analyzing Global CN {global_cn}...")
        
        # Get data for this global CN
        gcn_data = patient_densities[patient_densities['GlobalCN'] == global_cn]
        
        # Prepare data by dataset
        dataset_data = {}
        dataset_stats = {}
        
        for dataset in datasets:
            data = gcn_data[gcn_data['Dataset'] == dataset]['Density_LocalCNs_per_mm2'].values
            dataset_data[dataset] = data
            dataset_stats[dataset] = {
                'n': len(data),
                'mean': np.mean(data) if len(data) > 0 else 0,
                'median': np.median(data) if len(data) > 0 else 0,
                'std': np.std(data) if len(data) > 0 else 0
            }
        
        # Only test if we have data from multiple datasets
        datasets_with_data = [d for d in datasets if len(dataset_data[d]) > 0]
        
        if len(datasets_with_data) >= 2:
            # Kruskal-Wallis test if more than 2 datasets
            if len(datasets_with_data) > 2:
                try:
                    h_stat, p_kruskal = kruskal(*[dataset_data[d] for d in datasets_with_data])
                except:
                    h_stat, p_kruskal = np.nan, np.nan
            else:
                h_stat, p_kruskal = np.nan, np.nan
            
            # Pairwise Mann-Whitney U tests
            pairwise_results = []
            for dataset1, dataset2 in combinations(datasets_with_data, 2):
                data1 = dataset_data[dataset1]
                data2 = dataset_data[dataset2]
                
                if len(data1) > 0 and len(data2) > 0:
                    try:
                        u_stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                    except:
                        u_stat, p_value = np.nan, np.nan
                    
                    pairwise_results.append({
                        'GlobalCN': global_cn,
                        'Dataset1': dataset1,
                        'Dataset2': dataset2,
                        'U_statistic': u_stat,
                        'p_value': p_value,
                        'n1': len(data1),
                        'n2': len(data2),
                        'mean1': dataset_stats[dataset1]['mean'],
                        'mean2': dataset_stats[dataset2]['mean'],
                        'median1': dataset_stats[dataset1]['median'],
                        'median2': dataset_stats[dataset2]['median']
                    })
            
            # Add to all results
            for result in pairwise_results:
                result['Kruskal_H'] = h_stat
                result['Kruskal_p'] = p_kruskal
                all_results.append(result)
    
    # Convert to DataFrame and apply multiple testing correction
    if all_results:
        results_df = pd.DataFrame(all_results)
        
        # Apply FDR correction to p-values
        valid_p = results_df['p_value'].notna()
        corrected_p = np.full(len(results_df), np.nan)
        
        if valid_p.sum() > 0:
            corrected_p[valid_p] = multipletests(results_df['p_value'][valid_p], method='fdr_bh')[1]
        
        results_df['p_corrected'] = corrected_p
        results_df['significant'] = results_df['p_corrected'] < 0.05
        
        # Save results
        results_file = os.path.join(output_dir, "statistical_analysis_per_global_CN_TISSUE.csv")
        results_df.to_csv(results_file, index=False)
        
        # Create summary
        summary_file = os.path.join(output_dir, "statistical_analysis_per_global_CN_summary_TISSUE.txt")
        with open(summary_file, 'w') as f:
            f.write("STATISTICAL ANALYSIS PER GLOBAL CN (TISSUE MASK)\n")
            f.write("="*60 + "\n\n")
            
            for global_cn in sorted(results_df['GlobalCN'].unique()):
                gcn_results = results_df[results_df['GlobalCN'] == global_cn]
                f.write(f"Global CN {global_cn}:\n")
                f.write("-" * 20 + "\n")
                
                for _, row in gcn_results.iterrows():
                    sig = "YES" if row['significant'] else "NO"
                    f.write(f"  {row['Dataset1']} vs {row['Dataset2']}:\n")
                    f.write(f"    Raw p-value: {row['p_value']:.6f}\n")
                    f.write(f"    Corrected p-value (FDR): {row['p_corrected']:.6f}\n")
                    f.write(f"    Significant: {sig}\n")
                    f.write(f"    Means: {row['mean1']:.4f} vs {row['mean2']:.4f}\n")
                    f.write(f"    Sample sizes: {row['n1']} vs {row['n2']}\n")
                f.write("\n")
        
        print(f"📊 Statistical results (tissue mask) saved to: {results_file}")
        print(f"📄 Statistical summary (tissue mask) saved to: {summary_file}")
        
        # Print significant results
        significant_results = results_df[results_df['significant'] == True]
        if len(significant_results) > 0:
            print(f"\n🎯 SIGNIFICANT DIFFERENCES FOUND:")
            for _, row in significant_results.iterrows():
                print(f"  Global CN {row['GlobalCN']}: {row['Dataset1']} vs {row['Dataset2']} (p={row['p_corrected']:.4f})")
        else:
            print(f"\n⚠️  No significant differences found after multiple testing correction")
        
        return results_df
    else:
        print("❌ No statistical comparisons could be performed")
        return None

# UPDATED: Visualization functions with tissue area labels
def create_heatmap_average_density_by_dataset(density_df, output_dir):
    """
    Create heatmap of average density per global CN by dataset
    """
    print("🎨 Creating heatmap: Average density by dataset...")
    
    plt.figure(figsize=(16, 10))
    
    # Calculate average density per global CN by dataset
    avg_density_by_dataset = density_df.groupby(['Dataset', 'GlobalCN'])['Density_LocalCNs_per_mm2'].mean().reset_index()
    heatmap_data = avg_density_by_dataset.pivot(index='Dataset', columns='GlobalCN', values='Density_LocalCNs_per_mm2').fillna(0)
    
    if len(heatmap_data) > 0:
        sns.heatmap(heatmap_data, 
                    annot=True, 
                    fmt='.3f', 
                    cmap='Reds',
                    cbar_kws={'label': 'Avg Local CNs per mm² (Tissue Area)'},
                    linewidths=0.5,    # UPDATED: Thin black borders
                    linecolor='black', # UPDATED: Explicit black color
                    square=True)       # UPDATED: Square boxes
        plt.title('Average Local CN Density per Global CN by Dataset\n(Patient Level - Tissue Area)', fontsize=16, fontweight='bold')
        plt.xlabel('Global CN', fontsize=14)
        plt.ylabel('Dataset', fontsize=14)
    else:
        plt.text(0.5, 0.5, 'No data available for heatmap', ha='center', va='center', 
                transform=plt.gca().transAxes, fontsize=16)
        plt.title('Dataset Heatmap - No Data', fontsize=16)
    
    plt.tight_layout()
    file_path = os.path.join(output_dir, "01_heatmap_avg_density_by_dataset_PATIENT_LEVEL_TISSUE.png")
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"   Saved: {file_path}")

def create_heatmap_average_density_by_patient(density_df, output_dir):
    """
    Create heatmap of average density per global CN by patient
    """
    print("🎨 Creating heatmap: Average density by patient...")
    
    plt.figure(figsize=(16, 12))
    
    # Calculate average density per global CN by patient
    if len(density_df['Patient'].unique()) > 1:
        avg_density_by_patient = density_df.groupby(['Patient', 'GlobalCN'])['Density_LocalCNs_per_mm2'].mean().reset_index()
        heatmap_data = avg_density_by_patient.pivot(index='Patient', columns='GlobalCN', values='Density_LocalCNs_per_mm2').fillna(0)
        
        sns.heatmap(heatmap_data, 
                    annot=False, 
                    fmt='.3f', 
                    cmap='Blues',
                    cbar_kws={'label': 'Avg Local CNs per mm² (Tissue Area)'},
                    linewidths=0.5,    # UPDATED: Thin black borders
                    linecolor='black', # UPDATED: Explicit black color
                    square=True)       # UPDATED: Square boxes
        plt.title('Average Local CN Density per Global CN by Patient\n(Patient Level - Tissue Area)', fontsize=16, fontweight='bold')
        plt.xlabel('Global CN', fontsize=14)
        plt.ylabel('Patient', fontsize=14)
        
        # Rotate patient labels if too many
        if len(density_df['Patient'].unique()) > 10:
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
    else:
        plt.text(0.5, 0.5, 'Insufficient patient-level data', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
        plt.title('Patient-level Analysis Not Available', fontsize=16)
        plt.axis('off')
    
    plt.tight_layout()
    file_path = os.path.join(output_dir, "02_heatmap_avg_density_by_patient_PATIENT_LEVEL_TISSUE.png")
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"   Saved: {file_path}")

def create_violinplot_density_by_dataset_and_global_cn(density_df, output_dir):
    """
    Create violin plot of average density by global CN for each dataset
    """
    print("🎨 Creating violin plot: Average density by dataset and global CN...")
    
    plt.figure(figsize=(24, 12))
    
    # Use consistent dataset colors
    dataset_colors = {dataset: get_dataset_color(dataset) for dataset in density_df['Dataset'].unique()}
    
    global_cns = sorted(density_df['GlobalCN'].unique())
    datasets = sorted(density_df['Dataset'].unique())
    
    if len(datasets) > 1 and len(global_cns) > 0:
        # Prepare data for violin plot
        plot_data = []
        for dataset in datasets:
            for global_cn in global_cns:
                subset = density_df[(density_df['Dataset'] == dataset) & 
                                  (density_df['GlobalCN'] == global_cn)]
                if len(subset) > 0:
                    for _, row in subset.iterrows():
                        plot_data.append({
                            'Dataset': dataset,
                            'GlobalCN': f'GCN {int(global_cn)}',
                            'Density': row['Density_LocalCNs_per_mm2'],
                            'GlobalCN_num': global_cn
                        })
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            
            # Create violin plot
            ax = sns.violinplot(
                data=plot_df,
                x='GlobalCN',
                y='Density',
                hue='Dataset',
                palette=[dataset_colors.get(ds, f'C{i}') for i, ds in enumerate(datasets)],
                inner='box',  # Show box plot inside violin
                scale='width',  # Scale violins to have same width
                cut=0  # Don't extend beyond data range
            )
            
            # Add individual patient points with slight jitter
            for i, dataset in enumerate(datasets):
                dataset_data = plot_df[plot_df['Dataset'] == dataset]
                if len(dataset_data) > 0:
                    # Create jittered x positions for each global CN
                    for j, global_cn in enumerate(sorted(plot_df['GlobalCN_num'].unique())):
                        gcn_data = dataset_data[dataset_data['GlobalCN_num'] == global_cn]
                        if len(gcn_data) > 0:
                            # Calculate x position with offset for different datasets
                            n_datasets = len(datasets)
                            width = 0.8 / n_datasets  # Total width divided by number of datasets
                            offset = (i - (n_datasets - 1) / 2) * width * 0.6  # Position within group
                            x_pos = j + offset
                            
                            # Add jitter to x position
                            jitter = np.random.normal(0, 0.02, len(gcn_data))
                            x_positions = np.full(len(gcn_data), x_pos) + jitter
                            
                            plt.scatter(x_positions, gcn_data['Density'], 
                                      color='black', s=20, alpha=0.6, zorder=3)
            
            plt.title('Local CN Density Distribution by Global CN and Dataset\n(Patient Level - Tissue Area)\nViolin plots show density distribution, dots show individual patients', 
                      fontsize=16, fontweight='bold')
            plt.xlabel('Global CN', fontsize=14)
            plt.ylabel('Local CNs per mm² (Tissue Area)', fontsize=14)
            
            # Set y-axis to start from 0 (remove negative values)
            y_min, y_max = plt.ylim()
            plt.ylim(0, y_max)
            
            # Rotate x-axis labels if needed
            plt.xticks(rotation=45)
            plt.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3, axis='y')
            
        else:
            plt.text(0.5, 0.5, 'No data available for violin plot', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            plt.title('Insufficient Data for Violin Plot', fontsize=16)
            plt.axis('off')
            
    else:
        plt.text(0.5, 0.5, f'Insufficient data for comparison\n{len(datasets)} datasets, {len(global_cns)} global CNs', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.title('Insufficient Data for Comparison', fontsize=16)
        plt.axis('off')
    
    plt.tight_layout()
    file_path = os.path.join(output_dir, "04_violinplot_density_by_dataset_and_global_cn_PATIENT_LEVEL_TISSUE.png")
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"   Saved: {file_path}")

def create_violinplot_density_by_global_cn(density_df, output_dir):
    """
    Create violin plot of density distribution by global CN (all datasets combined)
    """
    print("🎨 Creating violin plot: Density distribution by global CN...")
    
    plt.figure(figsize=(20, 12))
    
    global_cns = sorted(density_df['GlobalCN'].unique())
    
    if len(global_cns) > 0:
        # Prepare data for violin plot
        plot_data = []
        for global_cn in global_cns:
            subset = density_df[density_df['GlobalCN'] == global_cn]
            for _, row in subset.iterrows():
                plot_data.append({
                    'GlobalCN': f'GCN {int(global_cn)}',
                    'Density': row['Density_LocalCNs_per_mm2'],
                    'GlobalCN_num': global_cn,
                    'Dataset': row['Dataset']
                })
        
        if plot_data:
            plot_df = pd.DataFrame(plot_data)
            
            # Sort by median density for better visualization
            median_densities = plot_df.groupby('GlobalCN_num')['Density'].median().sort_values(ascending=False)
            sorted_gcns = [f'GCN {int(gcn)}' for gcn in median_densities.index]
            
            # Create violin plot
            ax = sns.violinplot(
                data=plot_df,
                x='GlobalCN',
                y='Density',
                order=sorted_gcns,
                palette='viridis',
                inner='box',  # Show box plot inside violin
                scale='width',  # Scale violins to have same width
                cut=0  # Don't extend beyond data range
            )
            
            # Add individual patient points with jitter
            for i, gcn_label in enumerate(sorted_gcns):
                gcn_num = int(gcn_label.split()[1])
                gcn_data = plot_df[plot_df['GlobalCN_num'] == gcn_num]
                if len(gcn_data) > 0:
                    jitter = np.random.normal(0, 0.05, len(gcn_data))
                    x_positions = np.full(len(gcn_data), i) + jitter
                    plt.scatter(x_positions, gcn_data['Density'], 
                              color='black', s=20, alpha=0.7, zorder=3)
            
            plt.title('Local CN Density Distribution by Global CN\n(Patient Level - Tissue Area)\nViolin plots show density distribution, dots show individual patients', 
                      fontsize=16, fontweight='bold')
            plt.xlabel('Global CN (Ranked by Median Density)', fontsize=14)
            plt.ylabel('Local CNs per mm² (Tissue Area)', fontsize=14)
            
            # Set y-axis to start from 0 (remove negative values)
            y_min, y_max = plt.ylim()
            plt.ylim(0, y_max)
            
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            
        else:
            plt.text(0.5, 0.5, 'No data available for violin plot', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
            plt.title('No Data Available', fontsize=16)
            plt.axis('off')
    else:
        plt.text(0.5, 0.5, 'No global CNs found', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.title('No Global CNs Found', fontsize=16)
        plt.axis('off')
    
    plt.tight_layout()
    file_path = os.path.join(output_dir, "03_violinplot_density_by_global_cn_PATIENT_LEVEL_TISSUE.png")
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"   Saved: {file_path}")

def create_barplot_summary_by_dataset(density_df, output_dir):
    """
    Create summary bar plot by dataset with error bars and individual patient points
    """
    print("🎨 Creating bar plot: Summary by dataset...")
    
    plt.figure(figsize=(12, 10))
    
    datasets = density_df['Dataset'].unique()
    dataset_colors = {dataset: get_dataset_color(dataset) for dataset in datasets}
    
    if len(datasets) > 1:
        # Calculate total density per patient first
        patient_totals = density_df.groupby(['Dataset', 'Patient'])['Density_LocalCNs_per_mm2'].sum().reset_index()
        
        # Calculate dataset summary statistics
        dataset_summary = patient_totals.groupby('Dataset')['Density_LocalCNs_per_mm2'].agg(['mean', 'std', 'count']).reset_index()
        dataset_summary['se'] = dataset_summary['std'] / np.sqrt(dataset_summary['count'])
        
        # Additional summary info
        dataset_info = patient_totals.groupby('Dataset').agg({
            'Patient': 'nunique'
        }).reset_index()
        dataset_summary = dataset_summary.merge(dataset_info, on='Dataset')
        
        # Plot mean density by dataset with consistent colors and error bars
        bars = plt.bar(dataset_summary['Dataset'], dataset_summary['mean'], 
                       yerr=dataset_summary['se'],
                       color=[dataset_colors.get(ds, f'C{i}') for i, ds in enumerate(dataset_summary['Dataset'])],
                       alpha=0.8,
                       capsize=5,
                       error_kw={'linewidth': 2, 'alpha': 0.8})
        
        # Add individual patient data points for each dataset
        for i, dataset in enumerate(dataset_summary['Dataset']):
            individual_data = patient_totals[patient_totals['Dataset'] == dataset]['Density_LocalCNs_per_mm2'].values
            x_jitter = np.random.normal(i, 0.04, size=len(individual_data))
            plt.scatter(x_jitter, individual_data, color='black', s=30, alpha=0.7, zorder=3,
                       label='Individual Patients' if i == 0 else "")
        
        plt.title('Average Total Local CN Density by Dataset\n(Patient Level - Tissue Area)\nBars show mean ± SE, dots show individual patients', 
                  fontsize=16, fontweight='bold')
        plt.xlabel('Dataset', fontsize=14)
        plt.ylabel('Total Local CNs per mm² (Tissue Area)', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add patient counts as text
        for i, (_, row) in enumerate(dataset_summary.iterrows()):
            plt.text(i, 0.01, f'{int(row["count"])} patients', 
                    ha='center', va='bottom', fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    else:
        plt.text(0.5, 0.5, 'Only one dataset available', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=16)
        plt.title('Single Dataset Analysis', fontsize=16)
        plt.axis('off')
    
    plt.tight_layout()
    file_path = os.path.join(output_dir, "05_barplot_summary_by_dataset_PATIENT_LEVEL_TISSUE.png")
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"   Saved: {file_path}")

def create_statistics_summary_plot(density_df, stats_results, output_dir):
    """
    Create a text-based summary plot with statistics
    """
    print("🎨 Creating statistics summary...")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    # Calculate summary statistics
    total_patients = len(density_df['Patient'].unique())
    total_global_cns = len(density_df['GlobalCN'].unique())
    total_datasets = len(density_df['Dataset'].unique())
    avg_density_overall = density_df['Density_LocalCNs_per_mm2'].mean()
    std_density_overall = density_df['Density_LocalCNs_per_mm2'].std()
    max_density = density_df['Density_LocalCNs_per_mm2'].max()
    min_density = density_df['Density_LocalCNs_per_mm2'].min()
    
    # Get patient totals for overall statistics
    patient_totals = density_df.groupby(['Dataset', 'Patient'])['Density_LocalCNs_per_mm2'].sum().reset_index()
    
    stats_text = f"""PATIENT-LEVEL Local CN Density Analysis Summary (TISSUE AREA):

🎯 KEY UPDATES APPLIED:
✅ TISSUE MASK: Using tissue area instead of inflammation area
✅ Mask threshold: > 0 (all tissue regions)
✅ Patient-level aggregation: Each dot = one patient
✅ Consistent dataset colors: {list(set(density_df['Dataset'].unique()))}
✅ Error bars (standard error) on all bar plots
✅ Individual patient points overlaid on all bars
✅ Statistical analysis performed at patient level
✅ Split panels for better visibility
✅ Pixel conversion: 1 pixel = 0.1507 µm
✅ Area calculation: Based on tissue regions (all pixels > 0)

📊 ANALYSIS RESULTS:
📊 Datasets Successfully Analyzed: {total_datasets}
📊 Datasets Found: {', '.join(density_df['Dataset'].unique())}
👥 Total Patients: {total_patients}
🌐 Total Global CNs: {total_global_cns}

📈 Patient-Level Density Statistics (Local CNs per mm² Tissue Area):
  Mean: {avg_density_overall:.4f}
  Std Dev: {std_density_overall:.4f}
  Min: {min_density:.4f}
  Max: {max_density:.4f}

📋 Dataset Breakdown (Patient Level - Tissue Area):
"""
    
    for dataset in density_df['Dataset'].unique():
        dataset_patients = len(density_df[density_df['Dataset'] == dataset]['Patient'].unique())
        dataset_records = len(density_df[density_df['Dataset'] == dataset])
        dataset_avg_density = patient_totals[patient_totals['Dataset'] == dataset]['Density_LocalCNs_per_mm2'].mean()
        rgb_color = get_dataset_color(dataset)
        stats_text += f"  {dataset}: {dataset_patients} patients, {dataset_records} records (RGB: {rgb_color})\n"
        stats_text += f"    Avg total density per patient: {dataset_avg_density:.4f}\n"
    
    # Add statistical results if available
    if stats_results and 'pairwise_total_density' in stats_results:
        stats_text += f"\n🧮 STATISTICAL COMPARISONS (Patient Level - Tissue Area):\n"
        for result in stats_results['pairwise_total_density']:
            significance = "SIGNIFICANT" if result['significant'] else "Not significant"
            stats_text += f"  {result['dataset1']} vs {result['dataset2']}: {significance}\n"
            stats_text += f"    p-corrected = {result['p_corrected']:.4f}\n"
            stats_text += f"    Medians: {result['median1']:.4f} vs {result['median2']:.4f}\n"
    
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace')
    ax.axis('off')
    ax.set_title('Patient-Level Analysis Statistics Summary (Tissue Area)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    file_path = os.path.join(output_dir, "06_statistics_summary_PATIENT_LEVEL_TISSUE.png")
    plt.savefig(file_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"   Saved: {file_path}")

# NEW: Main function to create all split visualizations
def create_patient_level_split_visualizations(density_df, stats_results, output_dir):
    """
    Create all visualizations as separate files for patient-level analysis with violin plots using tissue area
    """
    print("🎨 Creating PATIENT-LEVEL split visualizations with violin plots (TISSUE AREA)...")
    print(f"📊 Input data: {len(density_df)} records from {len(density_df['Patient'].unique())} patients")
    
    # Print patient count verification
    print(f"\n📊 PATIENT COUNT VERIFICATION:")
    for dataset in density_df['Dataset'].unique():
        n_patients = len(density_df[density_df['Dataset'] == dataset]['Patient'].unique())
        patients = sorted(density_df[density_df['Dataset'] == dataset]['Patient'].unique())
        print(f"  {dataset}: {n_patients} patients - {patients}")
    
    # Print global CN coverage verification
    print(f"\n📊 GLOBAL CN COVERAGE VERIFICATION:")
    for global_cn in sorted(density_df['GlobalCN'].unique()):
        print(f"  Global CN {global_cn}:")
        for dataset in density_df['Dataset'].unique():
            subset = density_df[(density_df['Dataset'] == dataset) & (density_df['GlobalCN'] == global_cn)]
            n_patients = len(subset)
            zero_density = len(subset[subset['Density_LocalCNs_per_mm2'] == 0])
            nonzero_density = len(subset[subset['Density_LocalCNs_per_mm2'] > 0])
            print(f"    {dataset}: {n_patients} patients ({zero_density} with 0 density, {nonzero_density} with >0 density)")
    
    # Create all visualizations as separate files
    create_heatmap_average_density_by_dataset(density_df, output_dir)
    create_heatmap_average_density_by_patient(density_df, output_dir)
    create_violinplot_density_by_global_cn(density_df, output_dir)  # Changed to violin plot
    create_violinplot_density_by_dataset_and_global_cn(density_df, output_dir)  # Changed to violin plot
    create_barplot_summary_by_dataset(density_df, output_dir)  # Keep as bar plot for summary
    create_statistics_summary_plot(density_df, stats_results, output_dir)
    
    print("✅ All patient-level split visualizations with violin plots (tissue area) created successfully!")

# NEW: Function to analyze global CN frequency across datasets with comprehensive visualizations
def analyze_global_cn_frequency(global_cn_data, output_dir):
    """
    Analyze how frequently each global CN appears in each dataset with bar plots and heatmaps.
    """
    print("Analyzing global CN frequency across datasets...")
    
    # Count local CNs assigned to each global CN by dataset
    frequency_data = global_cn_data.groupby(['Dataset', 'GlobalCN']).size().reset_index(name='LocalCN_Count')
    
    # Pivot to create matrix: Dataset x GlobalCN
    frequency_matrix = frequency_data.pivot(index='Dataset', columns='GlobalCN', values='LocalCN_Count').fillna(0)
    
    # Calculate percentages within each dataset
    frequency_pct = frequency_matrix.div(frequency_matrix.sum(axis=1), axis=0) * 100
    
    # Create comprehensive figure with heatmaps and bar plots
    fig = plt.figure(figsize=(24, 20))
    
    # === HEATMAPS ===
    # Heatmap of counts - UPDATED WITH BORDERS
    ax1 = plt.subplot(3, 2, 1)
    sns.heatmap(frequency_matrix, 
                annot=False, 
                fmt='g', 
                cmap='Blues',
                cbar_kws={'label': 'Number of Local CNs'},
                linewidths=0.5,    # UPDATED: Thin black borders
                linecolor='black', # UPDATED: Explicit black color
                square=True,       # UPDATED: Square boxes
                ax=ax1)
    ax1.set_title('Number of Local CNs Assigned to Each Global CN by Dataset (Louvain)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Global CN', fontsize=12)
    ax1.set_ylabel('Dataset', fontsize=12)
    
    # Heatmap of percentages - UPDATED WITH BORDERS
    ax2 = plt.subplot(3, 2, 2)
    sns.heatmap(frequency_pct, 
                annot=False, 
                fmt='.1f', 
                cmap='Oranges',
                vmin=0,
                vmax=20,
                cbar_kws={'label': 'Percentage (%)'},
                linewidths=0.5,    # UPDATED: Thin black borders
                linecolor='black', # UPDATED: Explicit black color
                square=True,       # UPDATED: Square boxes
                ax=ax2)
    ax2.set_title('Percentage of Local CNs Assigned to Each Global CN by Dataset (Louvain)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Global CN', fontsize=12)
    ax2.set_ylabel('Dataset', fontsize=12)
    
    # === BAR PLOTS ===
    # Bar plot 1: Stacked bar chart by global CN
    ax3 = plt.subplot(3, 2, 3)
    dataset_colors_list = [get_dataset_color(ds) for ds in frequency_matrix.index]
    frequency_matrix.T.plot(kind='bar', stacked=True, ax=ax3, 
                           color=dataset_colors_list, alpha=0.8)
    ax3.set_title('Local CNs per Global CN by Dataset (Stacked)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Global CN', fontsize=12)
    ax3.set_ylabel('Number of Local CNs', fontsize=12)
    ax3.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Bar plot 2: Grouped bar chart by global CN
    ax4 = plt.subplot(3, 2, 4)
    frequency_matrix.T.plot(kind='bar', ax=ax4, 
                           color=dataset_colors_list, alpha=0.8)
    ax4.set_title('Local CNs per Global CN by Dataset (Grouped)', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Global CN', fontsize=12)
    ax4.set_ylabel('Number of Local CNs', fontsize=12)
    ax4.legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.tick_params(axis='x', rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # Bar plot 3: Stacked bar chart by dataset
    ax5 = plt.subplot(3, 2, 5)
    frequency_matrix.plot(kind='bar', stacked=True, ax=ax5, 
                         colormap='tab20', alpha=0.8)
    ax5.set_title('Global CN Distribution by Dataset (Stacked)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('Dataset', fontsize=12)
    ax5.set_ylabel('Number of Local CNs', fontsize=12)
    ax5.legend(title='Global CN', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    ax5.tick_params(axis='x', rotation=0)
    ax5.grid(True, alpha=0.3)
    
    # Bar plot 4: Percentage stacked bar chart by dataset
    ax6 = plt.subplot(3, 2, 6)
    frequency_pct.plot(kind='bar', stacked=True, ax=ax6, 
                      colormap='tab20', alpha=0.8)
    ax6.set_title('Global CN Distribution by Dataset (Percentage)', fontsize=14, fontweight='bold')
    ax6.set_xlabel('Dataset', fontsize=12)
    ax6.set_ylabel('Percentage (%)', fontsize=12)
    ax6.legend(title='Global CN', bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    ax6.tick_params(axis='x', rotation=0)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comprehensive frequency visualization
    freq_comprehensive_file = os.path.join(output_dir, "global_CN_frequency_comprehensive_louvain.png")
    plt.savefig(freq_comprehensive_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    # Save frequency data
    freq_counts_file = os.path.join(output_dir, "global_CN_frequency_counts_louvain.csv")
    frequency_matrix.to_csv(freq_counts_file)
    
    freq_pct_file = os.path.join(output_dir, "global_CN_frequency_percentages_louvain.csv")
    frequency_pct.to_csv(freq_pct_file)
    
    print(f"Comprehensive frequency visualization saved to: {freq_comprehensive_file}")
    print(f"Frequency data saved to: {freq_counts_file} and {freq_pct_file}")
    
    return frequency_matrix, frequency_pct

# ENHANCED: Improved function to visualize global CNs in tissue with separate plots per compartment
def visualize_global_cns_in_tissue(data_path, global_mapping, datasets, output_dir, epsilon, min_samples, filter_structural=True):
    """
    Visualize global CNs spatially distributed in tissue, creating separate figures for each compartment with legends.
    """
    print("Creating individual tissue visualizations of global CNs for each compartment...")
    
    # Load the main data
    dfInfo = pd.read_csv(data_path, low_memory=False)
    
    # Determine which cell type column to use
    cell_type_col = 'CellCatName3' if 'CellCatName3' in dfInfo.columns else 'MemCatName'
    
    for dataset in datasets:
        print(f"  Processing tissue visualizations for dataset: {dataset}")
        
        # Filter data for the current dataset
        dataset_data = dfInfo[dfInfo['Dataset'] == dataset]
        
        # Filter for immune cells if requested
        if filter_structural:
            dataset_data = filter_immune_cells_only(dataset_data, cell_type_col)
        
        # Get unique compartments
        compartments = dataset_data['CompName'].unique()
        print(f"    Found {len(compartments)} compartments: {list(compartments)}")
        
        # Create separate figure for each compartment
        for comp in compartments:
            print(f"    Creating visualization for compartment: {comp}")
            
            # Get compartment data
            comp_data = dataset_data[dataset_data['CompName'] == comp]
            
            if len(comp_data) == 0:
                print(f"      No data for {comp}, skipping...")
                continue
            
            # Load DBSCAN results for this compartment
            suffix = "_immune_only" if filter_structural else ""
            dbscan_dir = f'/project/mclark/SCAMPI_datasets/{dataset}/DBSCAN/DBSCAN_ep_{epsilon}_ms_{min_samples}{suffix}'
            dbscan_file = f'DBSCAN_{comp}_ms_{min_samples}_ep_{epsilon}.csv'
            dbscan_path = os.path.join(dbscan_dir, dbscan_file)
            
            try:
                # Read DBSCAN results
                dbscan_result = pd.read_csv(dbscan_path)
                
                # Merge with spatial data
                comp_data_with_cn = comp_data.merge(dbscan_result[['DBSCAN_label', 'CellID']], 
                                                   on='CellID', how='left')
                
                # Map local CNs to global CNs
                comp_data_with_cn['GlobalCN'] = comp_data_with_cn['DBSCAN_label'].apply(
                    lambda x: global_mapping.get((dataset, comp, int(x)), -1) if pd.notna(x) and x != -1 else -1
                )
                
                # Get unique global CNs (excluding noise)
                unique_global_cns = sorted([cn for cn in comp_data_with_cn['GlobalCN'].unique() if cn != -1])
                
                if len(unique_global_cns) == 0:
                    print(f"      No global CNs found for {comp}, skipping...")
                    continue
                
                # ENHANCED: Create larger figure with better layout
                fig, ax = plt.subplots(1, 1, figsize=(16, 12))
                
                # Create color mapping for global CNs
                colors = [get_consistent_cn_color(cn) for cn in unique_global_cns]
                cmap = mcolors.ListedColormap(['lightgray'] + colors)
                bounds = [-1.5] + [i - 0.5 for i in unique_global_cns] + [max(unique_global_cns) + 0.5 if unique_global_cns else 0.5]
                norm = mcolors.BoundaryNorm(bounds, cmap.N)
                
                # Create scatter plot
                scatter = ax.scatter(comp_data_with_cn['CellCentroidCol'],
                                   comp_data_with_cn['CellCentroidRow'],
                                   c=comp_data_with_cn['GlobalCN'],
                                   cmap=cmap,
                                   norm=norm,
                                   s=12,  # Slightly larger points
                                   alpha=0.8)
                
                # Set aspect ratio and flip axes for proper tissue orientation
                ax.invert_yaxis()
                ax.invert_xaxis()
                ax.set_aspect('equal')
                
                # ENHANCED: Add comprehensive legend
                global_cn_counts = comp_data_with_cn['GlobalCN'].value_counts()
                noise_count = global_cn_counts.get(-1, 0)
                
                # Create legend elements
                legend_elements = []
                
                # Add noise first
                if noise_count > 0:
                    legend_elements.append(
                        mpatches.Patch(color='lightgray', 
                                     label=f'Noise: {noise_count:,} cells')
                    )
                
                # Add global CNs sorted by cell count (descending)
                sorted_global_cns = sorted(unique_global_cns, 
                                         key=lambda x: global_cn_counts.get(x, 0), 
                                         reverse=True)
                
                # Limit legend entries to prevent overcrowding
                max_legend_entries = 20
                displayed_cns = sorted_global_cns[:max_legend_entries]
                
                for i, global_cn in enumerate(displayed_cns):
                    count = global_cn_counts.get(global_cn, 0)
                    color = get_consistent_cn_color(global_cn)
                    legend_elements.append(
                        mpatches.Patch(color=color, 
                                     label=f'Global CN {global_cn}: {count:,} cells')
                    )
                
                # Add truncation note if necessary
                if len(unique_global_cns) > max_legend_entries:
                    remaining_count = sum(global_cn_counts.get(cn, 0) 
                                        for cn in unique_global_cns[max_legend_entries:])
                    legend_elements.append(
                        mpatches.Patch(color='white', 
                                     label=f'... and {len(unique_global_cns) - max_legend_entries} more CNs ({remaining_count:,} cells)')
                    )
                
                # Position legend outside plot area
                legend = ax.legend(handles=legend_elements, 
                                 title=f'Global CNs in {comp} (Louvain)\n(Total: {len(comp_data_with_cn):,} cells)',
                                 bbox_to_anchor=(1.02, 1), 
                                 loc='upper left',
                                 fontsize=10,
                                 title_fontsize=12,
                                 frameon=True,
                                 fancybox=True,
                                 shadow=True)
                
                # ENHANCED: Improve title and labels
                immune_suffix = " (Immune Cells Only)" if filter_structural else ""
                ax.set_title(f'{dataset} - {comp} Global CNs Distribution (Louvain){immune_suffix}\n'
                           f'{len(unique_global_cns)} Global CNs, {len(comp_data_with_cn):,} Total Cells', 
                           fontsize=16, fontweight='bold', pad=20)
                
                ax.set_xlabel('Cell Centroid Col (μm)', fontsize=14, fontweight='bold')
                ax.set_ylabel('Cell Centroid Row (μm)', fontsize=14, fontweight='bold')
                
                # Save individual tissue visualization
                immune_suffix_file = "_immune_only" if filter_structural else ""
                tissue_viz_file = os.path.join(output_dir, 
                                             f"{dataset}_{comp}_global_CNs_tissue_louvain{immune_suffix_file}.png")
                plt.savefig(tissue_viz_file, bbox_inches="tight", dpi=300)
                plt.close()
                
                print(f"      Individual tissue visualization saved to: {tissue_viz_file}")
                
            except Exception as e:
                print(f"      Warning: Could not process {comp}: {e}")
                continue

# MODIFIED: Enhanced function to find global CNs using Louvain clustering with resolution control
def find_global_cns_with_louvain(composition_dfs, output_dir, focus_immune=True, include_cn_size=True, 
                                optimize_parameters=True, similarity_metric='cosine', similarity_threshold=0.3,
                                resolution=1.0, test_resolutions=True, create_manual_viz=True):
    """
    Find global CNs by clustering the CN compositions across all compartments and datasets
    using Louvain community detection algorithm with resolution control.
    
    Parameters:
    - resolution: Controls granularity. Higher values = more communities. 
                 Default: 1.0. Try 1.5-3.0 for detailed clustering.
    - test_resolutions: Whether to test multiple resolution values during optimization.
    """
    # Set environment variables to limit number of threads for numerical libraries
    os.environ['OPENBLAS_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['NUMEXPR_NUM_THREADS'] = '8'
    
    print("Combining composition data from all datasets...")
    # Combine all composition data
    combined_comp_df = pd.concat(composition_dfs, ignore_index=True)
    
    # Extract features for clustering (cell type proportions)
    all_features = [col for col in combined_comp_df.columns 
                   if col not in ['CN', 'CompName', 'TotalCells', 'Dataset']]
    
    # Filter for immune cell types if requested
    if focus_immune:
        feature_cols = [col for col in all_features if col in IMMUNE_CELL_TYPES]
        print(f"Focusing on {len(feature_cols)} immune cell types for clustering")
    else:
        feature_cols = all_features
        print(f"Using all {len(feature_cols)} cell types for clustering")
    
    if len(feature_cols) == 0:
        print("No cell type features found for clustering.")
        return None, None
    
    # Handle missing values
    combined_comp_df[feature_cols] = combined_comp_df[feature_cols].fillna(0)
    
    # Filter CNs with very few cells - using 6 cells as the threshold
    print("Filtering out CNs with fewer than 6 cells for downstream analysis.")
    combined_comp_df = combined_comp_df[combined_comp_df['TotalCells'] >= 6]
    
    # ENHANCED: Create feature matrix with CN size included
    print("Preparing data matrix for clustering...")
    
    # Start with cell type proportions
    X_proportions = combined_comp_df[feature_cols].values
    print(f"Cell type proportions matrix shape: {X_proportions.shape}")
    
    if include_cn_size:
        # Add CN size as an additional feature
        print("Including CN size as clustering factor...")
        
        # Log-transform CN sizes to reduce scale differences
        cn_sizes = np.log1p(combined_comp_df['TotalCells'].values).reshape(-1, 1)
        
        # Standardize features separately
        scaler_proportions = StandardScaler()
        X_proportions_scaled = scaler_proportions.fit_transform(X_proportions)
        
        scaler_sizes = StandardScaler()
        cn_sizes_scaled = scaler_sizes.fit_transform(cn_sizes)
        
        # Combine features
        X = np.hstack([X_proportions_scaled, cn_sizes_scaled])
        
        print(f"Combined feature matrix shape: {X.shape}")
        print(f"Features: {len(feature_cols)} cell type proportions + 1 CN size (log-transformed)")
        
        # Save scalers for later use
        scaler_info = {
            'proportions_scaler': scaler_proportions,
            'size_scaler': scaler_sizes,
            'feature_cols': feature_cols,
            'includes_size': True
        }
    else:
        # Use only cell type proportions
        X = X_proportions
        scaler_info = {
            'feature_cols': feature_cols,
            'includes_size': False
        }
        print(f"Using only cell type proportions matrix shape: {X.shape}")
    
    # Check data size
    data_size_mb = X.nbytes / (1024 * 1024)
    print(f"Data matrix size: {data_size_mb:.2f} MB")
    
    # Perform dimensionality reduction if the matrix is very large (optional)
    if X.shape[1] > 50:  # If we have more than 50 features
        from sklearn.decomposition import PCA
        print("Performing dimensionality reduction with PCA...")
        pca = PCA(n_components=min(50, X.shape[1]))
        X = pca.fit_transform(X)
        print(f"Reduced data matrix shape: {X.shape}")
        scaler_info['pca'] = pca
        scaler_info['pca_applied'] = True
    else:
        scaler_info['pca_applied'] = False
    
    # NEW: Optimize Louvain parameters including resolution if requested
    if optimize_parameters:
        print("Optimizing Louvain clustering parameters including resolution...")
        try:
            result = optimize_louvain_parameters(X, output_dir, test_resolutions=test_resolutions)
            
            # Handle different return formats
            if len(result) == 3:
                best_metric, best_threshold, best_resolution = result
            elif len(result) == 2:
                best_metric, best_threshold = result
                best_resolution = resolution if 'resolution' in locals() else 2.0  # Default higher resolution
            else:
                print("Unexpected optimization result format, using defaults")
                best_metric, best_threshold, best_resolution = 'cosine', 0.3, 1.0
            
            if best_metric is not None:
                similarity_metric = best_metric
                similarity_threshold = best_threshold
                resolution = best_resolution
                
        except Exception as e:
            print(f"Optimization failed: {e}")
            print("Using default parameters...")
            similarity_metric = 'cosine'
            similarity_threshold = 0.3
            resolution = resolution if 'resolution' in locals() else 2.0
    
    # Apply Louvain clustering with resolution
    print("Applying Louvain clustering to find global CNs...")
    global_cn_labels, modularity, graph_info = apply_louvain_clustering(
        X, similarity_metric=similarity_metric, similarity_threshold=similarity_threshold, 
        resolution=resolution, output_dir=output_dir
    )
    
    # Add global CN labels to the data
    print("Adding global CN labels to the data...")
    combined_comp_df['GlobalCN'] = global_cn_labels
    
    # Create mapping from local to global CNs
    print("Creating mapping from local to global CNs...")
    global_mapping = {}
    for idx, row in combined_comp_df.iterrows():
        dataset = row['Dataset']
        compname = row['CompName']
        local_cn = row['CN']
        global_cn = row['GlobalCN']
        global_mapping[(dataset, compname, local_cn)] = global_cn
    
    # Save the combined data with global CNs
    mapping_file = os.path.join(output_dir, "global_CN_mapping_louvain.csv")
    combined_comp_df.to_csv(mapping_file, index=False)
    print(f"Global CN mapping (Louvain) saved to: {mapping_file}")
    
    # Save the clustering information
    clustering_info = pd.DataFrame({
        'similarity_metric': [similarity_metric],
        'similarity_threshold': [similarity_threshold],
        'resolution': [resolution],
        'n_communities': [graph_info['n_communities']],
        'modularity': [modularity],
        'include_cn_size': [include_cn_size],
        'focus_immune': [focus_immune],
        'graph_density': [graph_info['density']],
        'avg_community_size': [graph_info['avg_community_size']],
        'std_community_size': [graph_info['std_community_size']],
        'method_used': [graph_info['method_used']]
    })
    clustering_info_file = os.path.join(output_dir, "louvain_clustering_info.csv")
    clustering_info.to_csv(clustering_info_file, index=False)
    print(f"Louvain clustering information saved to: {clustering_info_file}")
    
    # Save scaler information
    import pickle
    scaler_file = os.path.join(output_dir, "clustering_scalers_louvain.pkl")
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler_info, f)
    print(f"Clustering scalers (Louvain) saved to: {scaler_file}")
    
    # Create detailed cell type heatmap for each global CN
    print("Generating detailed cell type heatmap for each global CN...")
    global_cn_profiles = combined_comp_df.groupby('GlobalCN')[feature_cols].mean()
    
    # Get total cells in each global CN and average cells per CN
    global_cn_total_counts = combined_comp_df.groupby('GlobalCN')['TotalCells'].sum().astype(int)
    global_cn_avg_counts = combined_comp_df.groupby('GlobalCN')['TotalCells'].mean().astype(int)
    global_cn_num_instances = combined_comp_df.groupby('GlobalCN').size()
    
    # ENHANCED: Add size-based statistics
    global_cn_median_size = combined_comp_df.groupby('GlobalCN')['TotalCells'].median().astype(int)
    global_cn_size_std = combined_comp_df.groupby('GlobalCN')['TotalCells'].std().fillna(0).astype(int)
    
    # Save full profile data with additional statistics including size
    profile_data = global_cn_profiles.copy()
    profile_data['TotalCells'] = global_cn_total_counts
    profile_data['AvgCells'] = global_cn_avg_counts
    profile_data['MedianCells'] = global_cn_median_size
    profile_data['StdCells'] = global_cn_size_std
    profile_data['NumInstances'] = global_cn_num_instances
    
    profile_csv = os.path.join(output_dir, "global_CN_profiles_louvain.csv")
    profile_data.to_csv(profile_csv)
    
    # Create enhanced heatmap with size information
    plt.figure(figsize=(20, 16))
    
    # Cluster cell types (columns) by similarity
    col_linkage = linkage(global_cn_profiles.T, method='ward')
    
    # If we have many cell types, create a clustered heatmap
    if len(feature_cols) > 50:
        # Calculate variance of each cell type across global CNs
        variances = global_cn_profiles.var().sort_values(ascending=False)
        top_features = variances.index[:50].tolist()
        
        # Cluster the top cell types
        col_linkage = linkage(global_cn_profiles[top_features].T, method='ward')
        
        # Set custom yticklabels with cluster sizes, average cell counts, and median
        yticklabels = [f"Global CN {cn} (total={global_cn_total_counts[cn]}, avg={global_cn_avg_counts[cn]}, med={global_cn_median_size[cn]})" 
                      for cn in global_cn_profiles.index]
        
        # Create clustered heatmap with only the top variable features - UPDATED WITH BORDERS
        cg = sns.clustermap(
            global_cn_profiles[top_features],
            figsize=(20, 16),
            cmap="viridis",
            annot=False,
            fmt=".1f",
            linewidths=0.5,        # UPDATED: Thin black borders
            linecolor='black',     # UPDATED: Explicit black color
            col_linkage=col_linkage,
            col_cluster=True,
            row_cluster=False,
            dendrogram_ratio=(.1, .2),
            cbar_pos=(.02, .32, .03, .2),
            yticklabels=yticklabels
        )
        plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=12)
        plt.setp(cg.ax_heatmap.get_yticklabels(), fontsize=12)
        size_info = " + CN Size" if include_cn_size else ""
        plt.suptitle(f"Cell Type Composition (%) in {graph_info['n_communities']} Global CNs (Louvain R={resolution:.1f}{size_info}) (Top 50 Variable Cell Types)", fontsize=20, y=0.95)
    else:
        # Set custom yticklabels with cluster sizes, average cell counts, and median
        yticklabels = [f"Global CN {cn} (total={global_cn_total_counts[cn]}, avg={global_cn_avg_counts[cn]}, med={global_cn_median_size[cn]})" 
                      for cn in global_cn_profiles.index]
        
        # Create clustered heatmap with all features - UPDATED WITH BORDERS
        cg = sns.clustermap(
            global_cn_profiles,
            figsize=(20, 16),
            cmap="viridis",
            annot=False,
            fmt=".1f",
            linewidths=0.5,        # UPDATED: Thin black borders
            linecolor='black',     # UPDATED: Explicit black color
            col_linkage=col_linkage,
            col_cluster=True,
            row_cluster=False,
            dendrogram_ratio=(.1, .2),
            cbar_pos=(.02, .32, .03, .2),
            yticklabels=yticklabels
        )
        plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=12)
        plt.setp(cg.ax_heatmap.get_yticklabels(), fontsize=12)
        size_info = " + CN Size" if include_cn_size else ""
        plt.suptitle(f"Cell Type Composition (%) in {graph_info['n_communities']} Global CNs (Louvain R={resolution:.1f}{size_info})", fontsize=20, y=0.95)
    
    profiles_file = os.path.join(output_dir, "global_CN_profiles_heatmap_louvain.png")
    plt.savefig(profiles_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    # Create a non-clustered heatmap for easier interpretation
    plt.figure(figsize=(20, 16))
    if len(feature_cols) > 30:
        # Get top 30 most variable cell types
        variances = global_cn_profiles.var().sort_values(ascending=False)
        top_features = variances.index[:30].tolist()
        
        # Create heatmap with custom ytick labels showing cluster sizes and cell count stats - UPDATED WITH BORDERS
        ax = sns.heatmap(global_cn_profiles[top_features], 
                         cmap="viridis", 
                         annot=False, 
                         fmt=".1f", 
                         linewidths=0.5,        # UPDATED: Thin black borders
                         linecolor='black',     # UPDATED: Explicit black color
                         square=True)           # UPDATED: Square boxes
        
        # Create ytick labels with cluster sizes, average and median cell counts
        ytick_labels = [f"Global CN {int(cn)} (total={global_cn_total_counts[cn]}, avg={global_cn_avg_counts[cn]}, med={global_cn_median_size[cn]})" 
                      for cn in global_cn_profiles.index]
        ax.set_yticklabels(ytick_labels)
        
        size_info = " + CN Size" if include_cn_size else ""
        plt.title(f"Cell Type Composition (%) in {graph_info['n_communities']} Global CNs (Louvain R={resolution:.1f}{size_info}) (Top 30 Variable Cell Types)", fontsize=18)
    else:
        # Create heatmap with custom ytick labels showing cluster sizes and cell count stats - UPDATED WITH BORDERS
        ax = sns.heatmap(global_cn_profiles, 
                         cmap="viridis", 
                         annot=False, 
                         fmt=".1f", 
                         linewidths=0.5,        # UPDATED: Thin black borders
                         linecolor='black',     # UPDATED: Explicit black color
                         square=True)           # UPDATED: Square boxes
        
        # Create ytick labels with cluster sizes, average and median cell counts
        ytick_labels = [f"Global CN {int(cn)} (total={global_cn_total_counts[cn]}, avg={global_cn_avg_counts[cn]}, med={global_cn_median_size[cn]})" 
                      for cn in global_cn_profiles.index]
        ax.set_yticklabels(ytick_labels)
        
        size_info = " + CN Size" if include_cn_size else ""
        plt.title(f"Cell Type Composition (%) in {graph_info['n_communities']} Global CNs (Louvain R={resolution:.1f}{size_info})", fontsize=18)
    
    plt.ylabel("Global CN (total, avg, median cells)", fontsize=16)
    plt.xlabel("Cell Type", fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    
    basic_profiles_file = os.path.join(output_dir, "global_CN_profiles_basic_heatmap_louvain.png")
    plt.savefig(basic_profiles_file, bbox_inches="tight", dpi=300)
    plt.close()
    
    print(f"Global CN profiles (Louvain) saved to: {profile_csv}")
    
    # NEW: Create additional visualizations
    print("Creating additional visualizations...")
    
    # 1. Leave-one-out z-score heatmap
    loo_zscores = create_leave_one_out_zscore_heatmap(combined_comp_df, output_dir)
    
    # 2. Global CN frequency analysis
    frequency_matrix, frequency_pct = analyze_global_cn_frequency(combined_comp_df, output_dir)
    
    return global_mapping, combined_comp_df

# DBSCAN clustering function (unchanged)
def DBSCAN_spatial_segmentation(epsilon, min_samples, dfInfo, writeDir):
    """
    Perform DBSCAN clustering on spatial cell data.
    """
    dfToExport = pd.DataFrame(data=np.repeat(np.nan, dfInfo.shape[0]), columns=['DBSCAN_label'], index=list(dfInfo.index))
    clustering = DBSCAN(eps=epsilon, min_samples=min_samples).fit(dfInfo[['CellCentroidRow', 'CellCentroidCol']])
    for lab, idxTemp in zip(list(clustering.labels_), list(dfInfo.index)):
        dfToExport.loc[idxTemp] = lab
    dfToExport['CellID'] = dfInfo['CellID']
    dfToExport.to_csv(writeDir, index=True)
    print(f"DBSCAN clustering saved to: {writeDir}")
    return dfToExport

# Function to analyze cell type composition in each CN (now only immune cells) - UPDATED WITH BORDERS
def analyze_cn_composition(data, dbscan_results, compname, output_dir, focus_on_immune=True):
    """
    Analyze and visualize the composition of immune cell classes in each CN (DBSCAN cluster).
    """
    df = data[data['CompName'] == compname].copy()
    df = df.merge(dbscan_results[['DBSCAN_label', 'CellID']], on='CellID')
    
    # Determine which cell type column to use
    cell_type_col = 'CellCatName3' if 'CellCatName3' in df.columns else 'MemCatName'
    
    # Filter out noise points (DBSCAN_label = -1)
    df_clustered = df[df['DBSCAN_label'] != -1]
    
    if len(df_clustered) == 0:
        print(f"No clusters found in {compname}. Skipping composition analysis.")
        return None
    
    # Get unique clusters and cell types
    unique_clusters = sorted(df_clustered['DBSCAN_label'].unique())
    
    # Since we're now working with immune-filtered data, all cell types should be immune
    unique_cell_types = sorted(df_clustered[cell_type_col].unique())
    print(f"Found {len(unique_cell_types)} immune cell types in {compname}")
    
    # Initialize composition dataframe
    composition_data = []
    
    # Analyze each cluster
    for cluster in unique_clusters:
        cluster_df = df_clustered[df_clustered['DBSCAN_label'] == cluster]
        total_cells = len(cluster_df)
        
        # Initialize cluster data
        cluster_dict = {'CN': int(cluster), 'CompName': compname, 'TotalCells': total_cells}
        
        # Calculate proportions of each cell type
        cell_type_counts = cluster_df[cell_type_col].value_counts(normalize=True) * 100
        
        # Add proportions to cluster data
        for cell_type in unique_cell_types:
            cluster_dict[cell_type] = cell_type_counts.get(cell_type, 0)
        
        composition_data.append(cluster_dict)
    
    # Create composition DataFrame
    composition_df = pd.DataFrame(composition_data)
    
    # Save the composition data
    comp_file = os.path.join(output_dir, f"{compname}_immune_CN_composition.csv")
    composition_df.to_csv(comp_file, index=False)
    
    # Visualize the composition
    if len(unique_clusters) > 0:
        plt.figure(figsize=(14, 10))
        
        # Prepare data for heatmap
        heatmap_data = composition_df.set_index('CN')
        cell_counts = heatmap_data['TotalCells'].copy()
        
        # Drop non-cell type columns for heatmap
        if 'CompName' in heatmap_data.columns:
            heatmap_data = heatmap_data.drop(['CompName', 'TotalCells'], axis=1)
        
        # Create heatmap - UPDATED WITH BORDERS
        ax = sns.heatmap(heatmap_data, 
                        cmap="YlGnBu", 
                        annot=False, 
                        fmt=".1f", 
                        linewidths=0.5,    # UPDATED: Thin black borders
                        linecolor='black', # UPDATED: Explicit black color
                        square=True)       # UPDATED: Square boxes
        
        # Add total cell count as text annotations on the y-axis
        ax.set_yticks(np.arange(len(cell_counts)) + 0.5)
        ax.set_yticklabels([f"CN {int(cn)} (n={int(count)})" for cn, count in 
                          zip(heatmap_data.index, cell_counts)], rotation=0)
        
        plt.title(f"Immune Cell Type Composition (%) in CNs of {compname}", fontsize=16)
        plt.xlabel("Immune Cell Type", fontsize=12)
        plt.tight_layout()
        
        # Save heatmap
        heatmap_file = os.path.join(output_dir, f"{compname}_immune_CN_composition_heatmap.png")
        plt.savefig(heatmap_file, bbox_inches="tight", dpi=300)
        plt.close()
    
    print(f"Immune CN composition analysis saved to: {comp_file}")
    return composition_df

# Function to rename local CNs to global CNs (unchanged)
def rename_local_to_global_cns(data_path, dbscan_results_dir, global_mapping, dataset, output_dir):
    """
    Rename local CNs to global CNs in the original DBSCAN results.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all DBSCAN result files
    dbscan_files = [f for f in os.listdir(dbscan_results_dir) 
                   if os.path.isfile(os.path.join(dbscan_results_dir, f)) and f.endswith('.csv')]
    
    for dbscan_file in dbscan_files:
        # Extract compname from filename
        pattern = r"DBSCAN_([\w\d]+_Area\d+)_ms_(\d+)_ep_(\d+)"
        match = re.search(pattern, dbscan_file)
        
        if match:
            compname = match.group(1)
            
            # Read DBSCAN results
            dbscan_df = pd.read_csv(os.path.join(dbscan_results_dir, dbscan_file))
            
            # Create a new column for global CN
            dbscan_df['GlobalCN'] = dbscan_df['DBSCAN_label'].apply(
                lambda x: global_mapping.get((dataset, compname, int(x)), -1) if x != -1 else -1
            )
            
            # Save the renamed results
            output_file = os.path.join(output_dir, f"Global_Louvain_{dbscan_file}")
            dbscan_df.to_csv(output_file, index=False)
            
            print(f"Renamed CNs (Louvain) saved to: {output_file}")

# Updated main function - COMPLETE PIPELINE
def main():
    # Set environment variables to limit number of threads for numerical libraries
    os.environ['OPENBLAS_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['NUMEXPR_NUM_THREADS'] = '8'
    
    # Path to the combined dataset
    data_path = "/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/processed_combined_SAM_scores_filtered_alldataset5_10woMono_cellcatname23.csv"
    
    # Get user input for dataset selection
    print("Available datasets: IgG4, AAV, NK2")
    dataset_choice = input("Enter dataset to analyze (leave blank for all): ").strip()
    
    # Define datasets to process
    datasets = ['IgG4', 'AAV', 'NK2'] if not dataset_choice else [dataset_choice]
    
    # Process mode selection
    process_mode = input("Enter processing mode (1=DBSCAN only, 2=Patient-level analysis only, 3=Complete pipeline): ").strip() or "3"
    process_mode = int(process_mode)
    
    # Filter structural cells before DBSCAN
    filter_structural = input("Filter out structural/non-immune cells BEFORE DBSCAN? (y/n, default: y): ").strip().lower() != 'n'
    
    # DBSCAN parameters
    epsilon = 87
    min_samples = 6
    
    print("="*80)
    print("🔥 UPDATED: NOW USING TISSUE MASKS FOR DENSITY CALCULATION")
    print("="*80)
    print("Key changes:")
    print("- Tissue mask path: /project/mclark/SCAMPI_datasets/{Dataset}/tissue_composite_masks/DAPI_UV_[1|2]_{Dataset}_{Sample}.tif[f]")
    print("- Tissue threshold: > 0 (any tissue pixel)")
    print("- Pixel conversion: 1 pixel = 0.1507 µm")
    print("- Area calculation: Based on tissue regions only")
    print("="*80)
    
    # Enhanced clustering options for global CN analysis
    if process_mode in [2, 3]:
        # Option to include CN size in clustering
        include_cn_size = input("Include CN size as clustering factor? (y/n, default: y): ").strip().lower() != 'n'
        
        # Option to optimize Louvain parameters
        optimize_parameters = input("Optimize Louvain clustering parameters automatically? (y/n, default: y): ").strip().lower() != 'n'
        
        # Resolution parameter for controlling cluster granularity
        print("\nResolution parameter controls clustering granularity:")
        print("  - Lower values (0.5-1.0): Fewer, larger communities")
        print("  - Default (1.0): Balanced clustering")
        print("  - Higher values (1.5-3.0): More, smaller communities (detailed clustering)")
        
        if optimize_parameters:
            test_resolutions = input("Test multiple resolution values during optimization? (y/n, default: y): ").strip().lower() != 'n'
            resolution = 1.0  # Will be optimized
        else:
            test_resolutions = False
            resolution_input = input("Enter resolution parameter (default: 1.0, try 2.0-3.0 for detailed clusters): ").strip()
            resolution = float(resolution_input) if resolution_input else 1.0
        
        if not optimize_parameters:
            # Manual parameter selection
            print("Available similarity metrics: cosine, correlation")
            similarity_metric = input("Enter similarity metric (default: cosine): ").strip() or "cosine"
            similarity_threshold = input("Enter similarity threshold (default: 0.3): ").strip()
            similarity_threshold = float(similarity_threshold) if similarity_threshold else 0.3
            create_manual_viz = input("Create manual parameter analysis visualizations? (y/n, default: y): ").strip().lower() != 'n'
        else:
            similarity_metric = 'cosine'  # Will be optimized
            similarity_threshold = 0.3    # Will be optimized
            create_manual_viz = False  # Not applicable for auto-optimization
    
    # Focus on immune cells
    focus_immune = True
    print(f"Focusing on immune cells: {focus_immune}")
    print(f"Early structural cell filtering: {filter_structural}")
    
    # Process each dataset with DBSCAN (if requested)
    all_composition_dfs = []
    
    if process_mode in [1, 3]:
        print("\n🔍 RUNNING DBSCAN AND COMPOSITION ANALYSIS...")
        
        for dataset in datasets:
            print(f"\nProcessing dataset: {dataset}")
            
            # Create output directories
            suffix = "_immune_only" if filter_structural else ""
            dbscan_dir = f'/project/mclark/SCAMPI_datasets/{dataset}/DBSCAN/DBSCAN_ep_{epsilon}_ms_{min_samples}{suffix}'
            viz_dir = f'/project/mclark/SCAMPI_datasets/{dataset}/DBSCAN/viz_ep_{epsilon}_ms_{min_samples}{suffix}'
            comp_dir = f'/project/mclark/SCAMPI_datasets/{dataset}/DBSCAN/comp_ep_{epsilon}_ms_{min_samples}{suffix}'
            
            os.makedirs(dbscan_dir, exist_ok=True)
            os.makedirs(viz_dir, exist_ok=True)
            os.makedirs(comp_dir, exist_ok=True)
            
            # Load data for the current dataset
            print(f"Loading data for dataset: {dataset}")
            dfInfo = pd.read_csv(data_path, low_memory=False)
            dfInfo = dfInfo[dfInfo['Dataset'] == dataset]
            
            # Determine which cell type column to use
            cell_type_col = 'CellCatName3' if 'CellCatName3' in dfInfo.columns else 'MemCatName'
            
            # Get list of compartments
            comp_list = dfInfo['CompName'].unique()
            
            dataset_composition_dfs = []
            
            # Process each compartment
            for comp in comp_list:
                print(f"  Processing compartment: {comp}")
                
                # Filter data for the current compartment
                comp_data = dfInfo[dfInfo['CompName'] == comp]
                
                # Filter structural cells BEFORE DBSCAN if requested
                if filter_structural:
                    print(f"    Filtering structural/non-immune cells before DBSCAN...")
                    comp_data_filtered = filter_immune_cells_only(comp_data, cell_type_col)
                    
                    # Skip if no immune cells remain
                    if len(comp_data_filtered) < min_samples:
                        print(f"    Skipping {comp}: insufficient immune cells ({len(comp_data_filtered)} < {min_samples})")
                        continue
                else:
                    comp_data_filtered = comp_data.copy()
                
                # Perform DBSCAN on filtered data
                dbscan_output_path = f'{dbscan_dir}/DBSCAN_{comp}_ms_{min_samples}_ep_{epsilon}.csv'
                dbscan_result = DBSCAN_spatial_segmentation(
                    epsilon=epsilon, 
                    min_samples=min_samples, 
                    dfInfo=comp_data_filtered, 
                    writeDir=dbscan_output_path
                )
                
                # Analyze CN composition
                comp_df = analyze_cn_composition(
                    data=dfInfo if not filter_structural else pd.concat([comp_data_filtered], ignore_index=True),
                    dbscan_results=dbscan_result,
                    compname=comp,
                    output_dir=comp_dir,
                    focus_on_immune=focus_immune
                )
                
                if comp_df is not None:
                    # Add dataset column
                    comp_df['Dataset'] = dataset
                    dataset_composition_dfs.append(comp_df)
                    
                # Clear memory
                del comp_data
                del comp_data_filtered
                del dbscan_result
                gc.collect()
            
            # Combine all composition dfs for this dataset
            if dataset_composition_dfs:
                dataset_combined_df = pd.concat(dataset_composition_dfs, ignore_index=True)
                
                # Save combined composition for this dataset
                dataset_comp_file = os.path.join(comp_dir, f"{dataset}_all_immune_CN_composition.csv")
                dataset_combined_df.to_csv(dataset_comp_file, index=False)
                print(f"Combined composition for {dataset} saved to: {dataset_comp_file}")
                
                # Add to the list of all composition dataframes
                all_composition_dfs.append(dataset_combined_df)
            
            # Clear dataset-specific memory
            del dfInfo
            gc.collect()
    
    # Process global CNs and patient-level analysis with tissue masks
    if process_mode in [2, 3]:
        print("\n🔍 RUNNING GLOBAL CN ANALYSIS AND PATIENT-LEVEL DENSITY ANALYSIS (TISSUE MASKS)...")
        
        # Create output directory for global analysis
        suffix = "_louvain"
        if 'include_cn_size' in locals() and include_cn_size:
            suffix += "_with_size"
        if filter_structural:
            suffix += "_immune_only"
        suffix += "_tissue_mask"  # NEW: Add tissue mask suffix
        global_dir = f'/project/mclark/SCAMPI_datasets/AAV_NK2_IGG4_norm/Batch_Correction/global_CN_louvain1.5_0.5_analysis{suffix}'
        os.makedirs(global_dir, exist_ok=True)
        
        # Load composition data if not already available
        if process_mode == 2 and not all_composition_dfs:
            print("Loading composition data from files...")
            for dataset in datasets:
                comp_suffix = "_immune_only" if filter_structural else ""
                comp_dir = f'/project/mclark/SCAMPI_datasets/{dataset}/DBSCAN/comp_ep_{epsilon}_ms_{min_samples}{comp_suffix}'
                dataset_comp_file = os.path.join(comp_dir, f"{dataset}_all_immune_CN_composition.csv")
                
                try:
                    dataset_combined_df = pd.read_csv(dataset_comp_file)
                    all_composition_dfs.append(dataset_combined_df)
                    print(f"Loaded composition data for {dataset}")
                except Exception as e:
                    print(f"Error loading composition data for {dataset}: {e}")
        
        if all_composition_dfs:
            # Find global CNs using Louvain clustering
            print("Finding global CNs using Louvain clustering...")
            global_mapping, global_cn_data = find_global_cns_with_louvain(
                composition_dfs=all_composition_dfs,
                output_dir=global_dir,
                focus_immune=focus_immune,
                include_cn_size=include_cn_size if 'include_cn_size' in locals() else True,
                optimize_parameters=optimize_parameters if 'optimize_parameters' in locals() else True,
                similarity_metric=similarity_metric if 'similarity_metric' in locals() else 'cosine',
                similarity_threshold=similarity_threshold if 'similarity_threshold' in locals() else 0.3,
                resolution=resolution if 'resolution' in locals() else 1.0,
                test_resolutions=test_resolutions if 'test_resolutions' in locals() else True,
                create_manual_viz=create_manual_viz if 'create_manual_viz' in locals() else True
            )
            
            if global_mapping is not None:
                print("\n🎨 Creating tissue visualizations...")
                visualize_global_cns_in_tissue(
                    data_path=data_path,
                    global_mapping=global_mapping,
                    datasets=datasets,
                    output_dir=global_dir,
                    epsilon=epsilon,
                    min_samples=min_samples,
                    filter_structural=filter_structural
                )
                
                print("\n🔍 Starting PATIENT-LEVEL density analysis with TISSUE MASKS...")
                density_results, stats_results, per_gcn_stats = analyze_global_cn_density_patient_level(
                    global_cn_data=global_cn_data,
                    data_path=data_path,
                    datasets=datasets,
                    output_dir=global_dir,
                    epsilon=epsilon,
                    min_samples=min_samples,
                    filter_structural=filter_structural
                )

                if per_gcn_stats is not None:
                    print("\n" + "="*80)
                    print("PER-GLOBAL-CN STATISTICAL ANALYSIS RESULTS")
                    print("="*80)
        
                    # Show significant results
                    significant_results = per_gcn_stats[per_gcn_stats['significant'] == True]
        
                    if len(significant_results) > 0:
                        print(f"\n🎯 SIGNIFICANT DIFFERENCES FOUND FOR SPECIFIC GLOBAL CNs:")
                        for _, row in significant_results.iterrows():
                            print(f"  Global CN {row['GlobalCN']}: {row['Dataset1']} vs {row['Dataset2']}")
                            print(f"    Corrected p-value (FDR): {row['p_corrected']:.6f}")
                            print(f"    Mean densities: {row['mean1']:.4f} vs {row['mean2']:.4f}")
                            print(f"    Sample sizes (patients): {row['n1']} vs {row['n2']}")
                            print()
                    else:
                        print(f"\n⚠️  No significant differences found for individual Global CNs after multiple testing correction")
        
                    print(f"📊 Per-Global-CN statistical results saved to:")
                    print(f"    - {global_dir}/statistical_analysis_per_global_CN_TISSUE.csv")
                    print(f"    - {global_dir}/statistical_analysis_per_global_CN_summary_TISSUE.txt")
                    

                if density_results is not None:
                    # Create split visualizations with tissue area labels
                    print("\n🎨 Creating split visualizations with tissue area...")
                    create_patient_level_split_visualizations(
                        density_df=density_results,
                        stats_results=stats_results,
                        output_dir=global_dir
                    )
                    
                    # Print statistical results summary
                    if stats_results and 'pairwise_total_density' in stats_results:
                        print("\n" + "="*80)
                        print("PATIENT-LEVEL STATISTICAL ANALYSIS RESULTS")
                        print("="*80)
                        for result in stats_results['pairwise_total_density']:
                            significance = "YES" if result['significant'] else "NO"
                            print(f"{result['dataset1']} vs {result['dataset2']}:")
                            print(f"  Corrected p-value (FDR): {result['p_corrected']:.6f}")
                            print(f"  Significant: {significance}")
                            print(f"  Median densities: {result['median1']:.4f} vs {result['median2']:.4f}")
                            print(f"  Sample sizes (patients): {result['n1']} vs {result['n2']}")
                            print()
                
                # Rename local CNs to global CNs in each dataset
                for dataset in datasets:
                    print(f"\nRenaming local CNs to global CNs for dataset: {dataset}")
                    
                    dbscan_suffix = "_immune_only" if filter_structural else ""
                    dbscan_dir = f'/project/mclark/SCAMPI_datasets/{dataset}/DBSCAN/DBSCAN_ep_{epsilon}_ms_{min_samples}{dbscan_suffix}'
                    global_cn_dir = f'/project/mclark/SCAMPI_datasets/{dataset}/DBSCAN/global_CN_louvain_ep_{epsilon}_ms_{min_samples}{dbscan_suffix}'
                    
                    rename_local_to_global_cns(
                        data_path=data_path,
                        dbscan_results_dir=dbscan_dir,
                        global_mapping=global_mapping,
                        dataset=dataset,
                        output_dir=global_cn_dir
                    )
                
                print("\n" + "="*80)
                print("✅ TISSUE MASK ANALYSIS FINISHED!")
                print("="*80)
                print("KEY UPDATES IMPLEMENTED:")
                print("1. ✅ Tissue masks: Using tissue area instead of inflammation area")
                print("2. ✅ Pixel threshold > 0: Correctly identifying tissue regions")
                print("3. ✅ Patient-level aggregation: Data aggregated across compartments per patient")
                print("4. ✅ Split panels: 6 separate figures for better visibility")
                print("5. ✅ Individual patient dots: Each point represents one patient")
                print("6. ✅ Error bars: Standard error shown on all bar plots")
                print("7. ✅ Statistical rigor: Analysis performed at patient level")
                print("8. ✅ Consistent colors: Same dataset colors across all plots")
                print("9. ✅ Corrected pixel conversion: 1 pixel = 0.1507 µm")
                print("10. ✅ Tissue-specific density: CNs per mm² of tissue area")
                print("11. ✅ Square heatmap boxes with thin black borders")
                print(f"\n📁 All outputs saved to: {global_dir}")
                
                # Print parameter visualization info
                if 'optimize_parameters' in locals() and optimize_parameters:
                    print("\n🎯 PARAMETER OPTIMIZATION VISUALIZATIONS CREATED:")
                    print(f"    - {global_dir}/global_CN_parameter_optimization_explained.png")
                    print(f"    - {global_dir}/modularity_elbow_plot.png")
                    print(f"    - {global_dir}/louvain_parameter_optimization_with_resolution.csv")
                    
            else:
                print("\nFailed to find global CNs using Louvain clustering. Analysis incomplete.")
        else:
            print("\nNo composition data available. Please run DBSCAN analysis first.")
    
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()
