# 3.4 ) VISUALIZE CLUSTER - of NEOLOGISMS (result of generated KNOWLEDGE_MAPS)

print ("VISUALIZE NEOLOGISMS as CLUSTERS.")

import numpy as np
import json
from typing import List, Tuple, Dict, Any, Optional
from google import genai
from google.genai.errors import APIError
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans 
import plotly.express as px
import plotly.graph_objects as go
import ast
import re
import warnings

# Suppress harmless FutureWarnings from scikit-learn
warnings.filterwarnings('ignore', category=FutureWarning)

# --- COLOR CONSTANT DEFINITION ---
DEFAULT_COLOR_SEQUENCE = px.colors.qualitative.Bold
DEFAULT_CLUSTER_COLORS = {
    'Cluster 0': DEFAULT_COLOR_SEQUENCE[0],
    'Cluster 1': DEFAULT_COLOR_SEQUENCE[1],
    'Cluster 2': DEFAULT_COLOR_SEQUENCE[2],
    'Cluster 3': DEFAULT_COLOR_SEQUENCE[3],
    'Cluster 4': DEFAULT_COLOR_SEQUENCE[4],
    # Map the new cluster names to the colors
    'aCLUSTERZ_ATOMZ': DEFAULT_COLOR_SEQUENCE[0],
    'aCLUSTERZ_aXTRa': DEFAULT_COLOR_SEQUENCE[1],
    'aCLUSTERZ_aMECHZa': DEFAULT_COLOR_SEQUENCE[2],
    'aCLUSTERZ_aSOCIOa': DEFAULT_COLOR_SEQUENCE[3],
    'aCLUSTERZ_aMENTZa': DEFAULT_COLOR_SEQUENCE[4],
}

# --- SAME DATA AS ABOVE , but Mocked to ensure PERSISTENCE ---
MOCK_CLUSTER_JSON = {
    "aCLUSTERZ_ATOMZ": ["VIEWZ", "ACTZ", "CHOOZE", "VOIDZ"],
    "aCLUSTERZ_aXTRa": ["aXTRaCLARIa", "aXTRaFOCOa", "aXTRaWORDZa", "aXTRaVIEWZa"],
    "aCLUSTERZ_aMECHZa": ["aEXACTaMECHZa", "aCRAFTZaMECHZa", "aSELFaDESCRIPTZaMECHZa", "aSELFaHEALZaMECHZa"],
    "aCLUSTERZ_aSOCIOa": ["aSOCIOaCONFUZa", "aSOCIOaECHOZa", "aSOCIOaFLOWZa","aSOCIOaDYNAMICa"],
    "aCLUSTERZ_aMENTZa": ["aWIZDOaMENTZa", "aSPARKaMENTZa", "aDYNAMICaMENTZa", "aSTATICaMENTZa"],
}

# --- Data Loading and Fallback Logic ---
final_cluster_data: Dict[str, List[str]] = {}

try:
    cluster_data_raw = json.dumps(MOCK_CLUSTER_JSON)
    clean_string = cluster_data_raw.strip().lstrip("```json").lstrip().rstrip("```").strip()
    final_cluster_data = json.loads(clean_string)
    print("✅ VISUALIZE extra_dimensionality.")
    
except Exception as e:
    # This should now only catch JSON parsing errors if MOCK_CLUSTER_JSON was invalid
    print(f"🛑 CRITICAL ERROR: Could not parse MOCK_CLUSTER_JSON. {e}")

def prepare_cluster_labels(word_labels: List[str], cluster_data: Dict[str, List[str]]) -> List[str]:
    """
    Maps each word label to its designated cluster name from the loaded JSON data.
    """
    word_to_cluster = {}
    for cluster_name, word_list in cluster_data.items():
        for word in word_list:
            word_to_cluster[word] = cluster_name
            
    mapped_labels = []
    
    for word in word_labels:
        cluster_name = word_to_cluster.get(word)
        if cluster_name is None:
            mapped_labels.append('ERROR_NOT_FOUND')
        else:
            mapped_labels.append(cluster_name)
    
    return mapped_labels

# --- Configuration and Data Simulation ---
N_SAMPLES = 120 #20 # Only need enough samples to cover all words in the mock JSON
N_DIMENSIONS = 128
N_CLUSTERS = 5 # Should match the number of clusters in MOCK_CLUSTER_JSON
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

def generate_simulated_embeddings(cluster_data: Dict[str, List[str]], n_dim: int, total_samples: int) -> Tuple[np.ndarray, List[str]]:
    """
    Generates simulated embeddings by repeating and perturbing the core JSON words 
    to meet the required total_samples count for t-SNE stability.
    """
    
    all_words = []
    cluster_centers_map = {}
    cluster_names = list(cluster_data.keys())
    
    # 1. Create one center per cluster defined in the JSON
    cluster_centers = np.random.randn(len(cluster_names), n_dim) * 5
    cluster_map = {name: cluster_centers[i] for i, name in enumerate(cluster_names)}
    
    X = []
    labels = []
    
    # 2. Use the words from the JSON list in a repeating pattern to hit N_SAMPLES
    core_words_list = [word for word_list in cluster_data.values() for word in word_list]
    
    # Calculate how many times the core list needs to be repeated (at least once)
    num_repeats = max(1, total_samples // len(core_words_list)) + 1
    
    current_index = 0
    # Repeat the process until we have enough samples
    for _ in range(num_repeats):
        for cluster_name, word_list in cluster_data.items():
            center = cluster_map[cluster_name]
            for word in word_list:
                if len(X) < total_samples:
                    # Generate vector close to its cluster center with noise
                    vector = center + np.random.randn(n_dim) * 1.5
                    X.append(vector)
                    labels.append(word)
                else:
                    break
            if len(X) >= total_samples:
                break
        if len(X) >= total_samples:
            break
            
    print(f"Generated {len(X)} vectors from JSON list for t-SNE stability.")
    return np.array(X), labels

# 🛑 And update the call to this function:
X_high_dim, word_labels = generate_simulated_embeddings(final_cluster_data, N_DIMENSIONS, N_SAMPLES)
print(f"Simulated Data Shape: {X_high_dim.shape}")

# --- Dimensionality Reduction Functions (Unchanged) ---

def reduce_dimensions(X: np.ndarray, method: str = 'tsne', n_components: int = 3) -> Tuple[np.ndarray, PCA | None]:
    """Reduces the dimensionality of the embedding matrix X to n_components."""
    print(f"Starting {method.upper()} reduction to {n_components} dimensions...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=RANDOM_SEED, n_jobs=-1, init='pca', learning_rate='auto')
        X_reduced = reducer.fit_transform(X)
        return X_reduced, None
    elif method == 'pca':
        reducer = PCA(n_components=n_components, random_state=RANDOM_SEED)
        X_reduced = reducer.fit_transform(X)
        return X_reduced, reducer
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")

# --- Goal 1: Visualize High Dimensional Clusters (Dark Mode) ---

def visualize_clusters(X_high_dim: np.ndarray, word_labels: List[str], final_cluster_data: Dict[str, List[str]]):
    """
    Plots embeddings, using consistent colors and labeling from the hard-coded JSON data.
    """
    
    # 1. Determine Cluster Labels and Colors
    print("\n--- GOAL 1: Visualizing Loaded CLUSTERZ (Text Overlay Enabled) ---")
    
    df_cluster_labels = prepare_cluster_labels(word_labels, final_cluster_data)
    cluster_source = 'CLUSTERZ Label'
    
    custom_color_map = {}
    text_labels = []
    # Use the cluster names from the loaded data for iteration
    cluster_names = list(final_cluster_data.keys())
    
    for i, name in enumerate(cluster_names):
        if i < 5: 
            color_key = f'Cluster {i}'
            custom_color_map[name] = DEFAULT_CLUSTER_COLORS.get(color_key)
            text_labels.append(name) # Include all 5 labels 
        else: 
            custom_color_map[name] = 'gray' 
    
    cluster_colors = custom_color_map

    # 2. Reduce Dimensionality using t-SNE for visualization
    X_3d, _ = reduce_dimensions(X_high_dim, method='tsne', n_components=3)
    
    # 3. Create DataFrame for Plotly
    df = pd.DataFrame({
        'X': X_3d[:, 0],
        'Y': X_3d[:, 1],
        'Z': X_3d[:, 2],
        'Cluster': df_cluster_labels,
        'Word': word_labels
    })

    # Filter out any unexpected 'ERROR' labels if they occurred
    df_filtered = df[df['Cluster'] != 'ERROR_NOT_FOUND']
    
    # 4. Generate Interactive 3D Scatter Plot
    fig = px.scatter_3d(
        df_filtered, 
        x='X',
        y='Y',
        z='Z',
        color='Cluster', 
        color_discrete_map=cluster_colors,
        hover_name='Word', 
        title=f'Goal 1: Word Embedding Clusters by {cluster_source} (t-SNE)',
        opacity=0.9,
        height=700,
        template='plotly_dark',
    )
    
    # 5. AUGMENTATION: Add Cluster Text Overlays
    if text_labels:
        print(f"Adding text labels for: {', '.join(text_labels)}.")
        
        # Calculate the center point for each cluster used for text overlay
        try:
            # We use df_filtered since it contains only the points we want to plot
            cluster_centers_3d = df_filtered.groupby('Cluster')[['X', 'Y', 'Z']].mean().loc[text_labels]
        except KeyError as e:
             # This means the DataFrame is missing a cluster name from our text_labels list
            print(f"Warning: Could not calculate center for some clusters. Missing: {e}")
            # Filter text_labels to only those present in the filtered data
            text_labels = [label for label in text_labels if label in df_filtered['Cluster'].unique()]
            if not text_labels: return
            cluster_centers_3d = df_filtered.groupby('Cluster')[['X', 'Y', 'Z']].mean().loc[text_labels]
        Z_OFFSET = 44
        for cluster_name in text_labels:
            try:
                center = cluster_centers_3d.loc[cluster_name]
                # 🛑 APPLY OFFSET TO Z-COORDINATE
                offset_z = center['Z'] + Z_OFFSET                
                fig.add_trace(go.Scatter3d(
                    x=[center['X']], y=[center['Y']], 
                    z=[offset_z], # Use the offset Z coordinate
                    mode='text',
                    text=[cluster_name], 
                    textfont=dict(
                        color=cluster_colors.get(cluster_name, 'white'), 
                        size=16,
                        family="Arial Black" 
                    ),
                    textposition='top center',
                    name=cluster_name,
                    showlegend=False
                ))
            except KeyError:
                print(f"Skipping text for {cluster_name}: No data points found.")
                pass


    # 6. Final Layout Polish
    fig.update_layout(
        paper_bgcolor='black',
        plot_bgcolor='black',
        scene=dict(
            xaxis=dict(title='t-SNE 1', backgroundcolor='black', gridcolor='gray', showbackground=False),
            yaxis=dict(title='t-SNE 2', backgroundcolor='black', gridcolor='gray', showbackground=False),
            zaxis=dict(title='t-SNE 3', backgroundcolor='black', gridcolor='gray', showbackground=False),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend_title_text=cluster_source,
        font=dict(color='white')
    )
    
    # Show the plot
    fig.show()


# --- Execution ---
# 🛑 Final execution call simplified
visualize_clusters(X_high_dim, word_labels, final_cluster_data)

# SIGNIFICANCE: the opportunity to ADD ENHANCED_SYNTAX in this way is vast. 