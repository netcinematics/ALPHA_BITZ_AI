# 3.5 ) VISUALIZE VECTOR SPECTRUM DIMENSIONALITY - of EXACT_OPPOSITES!

print("VISUALIZE EXACT_OPPOSITES (as 'spectrum_vector') in embedded space.")
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Tuple, Dict

# --- Configuration and Data Simulation ---

N_SAMPLES = 500
N_DIMENSIONS = 128
N_CLUSTERS = 5
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# Define the specific words needed for Goal 2
TARGET_OPPOSITES = {
    "light": "dark",
    "fact": "false",
    "up": "down",
    "good": "bad"
}
TARGET_WORDS = list(TARGET_OPPOSITES.keys()) + list(TARGET_OPPOSITES.values())

def generate_simulated_embeddings(n_samples: int, n_dim: int, n_clusters: int, target_opposites: Dict[str, str]) -> Tuple[np.ndarray, List[str], Dict[str, np.ndarray]]:
    """
    Generates synthetic word embeddings, ensuring specific target words exist 
    and their opposites are positioned correctly for visualization.
    """
    
    X = []
    labels = []
    
    # Initialize the dictionary to store specific word vectors
    target_vectors = {}
    
    # 1. Generate random background words (Centered around origin)
    # We distribute them broadly so the vectors travel through a "cloud" of meaning
    for i in range(n_samples):
        vector = np.random.randn(n_dim) * 5
        X.append(vector)
        labels.append(f"bg_word_{i:03d}")

    # 2. Inject the target word pairs
    # We center these around the origin (0,0,0) to ensure they appear in the middle of the plot
    
    pair_index = 0
    for word_a, word_b in target_opposites.items():
        # Create a random direction for this concept pair
        direction = np.random.randn(n_dim)
        direction = direction / np.linalg.norm(direction) # Normalize
        
        # Magnitude of the vector (distance between opposites)
        magnitude = 8.0 
        
        # Calculate the center point for this pair (slightly offset per pair to avoid overlap)
        pair_center = np.random.randn(n_dim) * 2.0
        
        # Calculate positions relative to the pair center
        vector_a = pair_center + (direction * (magnitude / 2))
        vector_b = pair_center - (direction * (magnitude / 2))
        
        # Add to the data
        X.append(vector_a)
        labels.append(word_a)
        target_vectors[word_a] = vector_a
        
        X.append(vector_b)
        labels.append(word_b)
        target_vectors[word_b] = vector_b
        
        pair_index += 1

    return np.array(X), labels, target_vectors

X_high_dim, word_labels, target_vectors = generate_simulated_embeddings(N_SAMPLES, N_DIMENSIONS, N_CLUSTERS, TARGET_OPPOSITES)
print(f"Simulated Data Shape: {X_high_dim.shape}")
# ----------------------------------------------------------------

# --- Dimensionality Reduction Functions ---

def reduce_dimensions(X: np.ndarray, method: str = 'pca', n_components: int = 3) -> Tuple[np.ndarray, PCA | None]:
    """
    Reduces the dimensionality of the embedding matrix X to n_components.
    """
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


# --- Goal 2: Show Dimensionality Vectors of Opposites (Dark Mode) ---

def visualize_opposite_vectors(X_high_dim: np.ndarray, word_labels: List[str], target_opposites: Dict[str, str]):
    """
    Uses PCA to reduce dimensionality and plots the vectors connecting 
    pairs of opposite concepts in Dark Mode.
    """
    print("\n--- GOAL 2: Visualizing Opposing Concept Vectors (Dark Mode) ---")
    
    # 1. Reduce Dimensionality using PCA
    X_3d_pca, _ = reduce_dimensions(X_high_dim, method='pca', n_components=3)
    
    # 2. Create DataFrame for plotting ALL words
    df_all = pd.DataFrame({
        'X': X_3d_pca[:, 0],
        'Y': X_3d_pca[:, 1],
        'Z': X_3d_pca[:, 2],
        'Word': word_labels,
    })
    
    # 3. Determine types for coloring
    target_keys = list(target_opposites.keys())
    target_values = list(target_opposites.values())
    
    def get_type(word):
        if word in target_keys:
            return 'Positive Pole (Light/Fact)'
        elif word in target_values:
            return 'Negative Pole (Dark/False)'
        else:
            return 'Background'
    
    df_all['Type'] = df_all['Word'].apply(get_type)
    
    # Filter background to reduce noise but keep context
    df_targets = df_all[df_all['Type'] != 'Background']
    df_bg = df_all[df_all['Type'] == 'Background'].sample(n=100, random_state=RANDOM_SEED)
    df_plot = pd.concat([df_targets, df_bg])
    
    # 4. Define Colors (Dark Mode: Steelblue & Yellow)
    color_map = {
        'Positive Pole (Light/Fact)': 'steelblue', # Steelblue for one side
        'Negative Pole (Dark/False)': 'yellow',    # Yellow for the other
        'Background': 'rgba(100, 100, 100, 0.15)'  # Faint gray
    }

    # 5. Create the Base Scatter Plot
    fig = px.scatter_3d(
        df_plot,
        x='X',
        y='Y',
        z='Z',
        color='Type', 
        color_discrete_map=color_map,
        hover_name='Word',
        title='Goal 2: Conceptual Vectors of Opposites',
        opacity=0.9,
        size=[12 if t != 'Background' else 2 for t in df_plot['Type']],
        height=700,
        template='plotly_dark'
    )
    
    # 6. Add the Vector Traces (Lines connecting the opposites)
    for word_a, word_b in target_opposites.items():
        # Get the 3D coordinates for A and B
        coords_a = df_targets[df_targets['Word'] == word_a][['X', 'Y', 'Z']].values[0]
        coords_b = df_targets[df_targets['Word'] == word_b][['X', 'Y', 'Z']].values[0]
        
        # Add the line trace (vector)
        # Using a gradient-like effect by drawing a white line with opacity
        fig.add_trace(go.Scatter3d(
            x=[coords_a[0], coords_b[0]],
            y=[coords_a[1], coords_b[1]],
            z=[coords_a[2], coords_b[2]],
            mode='lines',
            line=dict(color='white', width=4), # White line stands out best on black
            name=f'{word_a} <-> {word_b}',
            hovertext=f'{word_a} to {word_b} vector',
            showlegend=False
        ))
        
        # Add text labels manually to control color
        fig.add_trace(go.Scatter3d(
            x=[coords_a[0]], y=[coords_a[1]], z=[coords_a[2]],
            mode='text',
            text=[word_a],
            textfont=dict(color='steelblue', size=12, family='Arial Black'),
            textposition='top center',
            showlegend=False
        ))
        fig.add_trace(go.Scatter3d(
            x=[coords_b[0]], y=[coords_b[1]], z=[coords_b[2]],
            mode='text',
            text=[word_b],
            textfont=dict(color='yellow', size=12, family='Arial Black'),
            textposition='top center',
            showlegend=False
        ))


    # 7. Final Layout Polish for "Dark Format"
    fig.update_layout(
        paper_bgcolor='black',
        plot_bgcolor='black',
        scene=dict(
            xaxis=dict(title='PCA 1', backgroundcolor='black', gridcolor='gray', showbackground=False),
            yaxis=dict(title='PCA 2', backgroundcolor='black', gridcolor='gray', showbackground=False),
            zaxis=dict(title='PCA 3', backgroundcolor='black', gridcolor='gray', showbackground=False),
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend_title_text='Pole Type',
        font=dict(color='white')
    )
    
    fig.show()


# --- Execution ---

# Execute the visualization for Goal 2
visualize_opposite_vectors(X_high_dim, word_labels, TARGET_OPPOSITES)