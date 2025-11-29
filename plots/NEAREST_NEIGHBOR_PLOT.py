# 3.2 ) NEAREST_NEIGHBOR 3D PLOT: show high-dimensional vectors of neologisms.

print("Nearest Neighbor - 'EXTRA_DIMENSIONALITY'.")
print("---------------------------------------------  run twice???")

# Suppress harmless FutureWarnings from scikit-learn
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    # Attempt to initialize client assuming it's correctly set up in Kaggle
    client = genai.Client()
    print("✅ genai client created.")
except Exception as e:
    # Placeholder for environment where client is not configured
    print(f"Warning: Could not initialize genai client. Ensure proper authentication: {e}")

# --- Embedding Model and Dimension Constants ---
EMBEDDING_MODEL = 'text-embedding-004' 
EMBEDDING_DIMENSIONS = 768
N_DIMENSIONS_REAL = EMBEDDING_DIMENSIONS

# --- AGENT AND TOOL DEFINITIONS ---
def get_word_embeddings(words: List[str]) -> List[List[float]]: 
    """
    Generates high-dimensional vector embeddings for a list of words.
    """
    print(f"🤖 Embedding Agent Tool: Requesting embeddings for {len(words)} words...")
    
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=words,
        )
        
        embedding_objects = response.embeddings
        embedding_list = [obj.values for obj in embedding_objects]
        
        print(f"✅ Embeddings generated. Returning as List[List[float]]...")
        return embedding_list
        
    except APIError as e:
        print(f"🛑 Embedding API Error: {e}")
        return [] 
    except Exception as e:
        print(f"🛑 Unexpected Error during embedding generation: {e}")
        return []

# ----------------------------------------------------------
# --- ISOLATED TEST EXECUTION (using GENAI CLIENT for stability) ---

# 🛑 Updated list of words
# TEST_WORDS = ["words", "WORDZ", "views", "VIEWZ"]
TEST_WORDS = ["words", "WORDZ", "views", "VIEWZ", "atoms", "ATOMZ"]

print(f"--- Starting Isolated Embedding Test for: {TEST_WORDS} ---")

# 🛑 DIRECT TOOL CALL
embedding_result_list = get_word_embeddings(TEST_WORDS)

# 3. Convert the clean list of lists result directly into a NumPy array
X_high_dim_test = np.empty((0, N_DIMENSIONS_REAL)) 
success = False

# Direct Conversion
if embedding_result_list:
    try:
        temp_array = np.array(embedding_result_list)
        
        # 4. Validation checks
        if temp_array.ndim >= 2 and temp_array.shape[0] == len(TEST_WORDS) and temp_array.shape[1] == N_DIMENSIONS_REAL:
            X_high_dim_test = temp_array
            success = True
        else:
            print(f"🛑 Conversion failed: Array shape {temp_array.shape} is not the expected ({len(TEST_WORDS)}, {N_DIMENSIONS_REAL}).")
            
    except Exception as e:
        print(f"🛑 ERROR during direct NumPy conversion: {e}")
        
if success:
    print("\n--- Test Results ---")
    print("✅ Successfully converted list data to NumPy array.")
    print(f"Shape of X_high_dim_test: {X_high_dim_test.shape}")
    print(f"Word Labels: {TEST_WORDS}")
else:
    print("\n--- Test Failed ---")
    print(f"X_high_dim_test remains empty with shape: {X_high_dim_test.shape}")


# ----------------------------------------------------------
# --- DATA INTEGRATION AND PLOTTING ---

# --- Configuration and Data Simulation ---
N_SAMPLES = 500
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 🛑 1. DEFINE REAL TARGETS (The full desired list)
COMMON_WORDS_REAL_FULL = ['words', 'views', 'atoms']
WORDZ_LIST_REAL_FULL = ['WORDZ', 'VIEWZ', 'ATOMZ']


def generate_simulated_embeddings(n_samples: int, n_dim: int, common_words: List[str], wordz_list: List[str]) -> Tuple[np.ndarray, List[str]]:
    # This function is retained but its output will be ignored for the final plot data
    X = []
    labels = []
    concepts_center = np.zeros(n_dim)
    void_center_a = np.random.randn(n_dim) * 5 
    void_center_b = void_center_a + np.ones(n_dim) * 8
    # ... (rest of simulation logic)
    return np.array(X), labels

# 🛑 2. RUN SIMULATION ONLY FOR BACKGROUND (Output is ignored below)
X_simulated_bg, labels_simulated_bg = generate_simulated_embeddings(
    N_SAMPLES, N_DIMENSIONS_REAL, [], []
)

# 🛑 3. COMBINE REAL DATA WITH BACKGROUND
X_REAL_TARGET = X_high_dim_test 

# Ensure the label list length matches the array length
N_REAL_VECTORS = X_REAL_TARGET.shape[0]
LABELS_REAL_TARGET = TEST_WORDS[:N_REAL_VECTORS] if N_REAL_VECTORS > 0 else []

# Adjust the plotting lists based on what was actually loaded 
N_PAIRS = N_REAL_VECTORS // 2
COMMON_WORDS_REAL = COMMON_WORDS_REAL_FULL[:N_PAIRS]
WORDZ_LIST_REAL = WORDZ_LIST_REAL_FULL[:N_PAIRS]

if N_REAL_VECTORS != len(TEST_WORDS):
    print(f"⚠️ Warning: Loaded {N_REAL_VECTORS} real vectors, expected {len(TEST_WORDS)}. Plotting only the successfully loaded pairs.")

# 🛑 CORE CHANGE: ONLY USE REAL TARGET EMBEDDINGS (Remove background)
X_combined = X_REAL_TARGET
word_labels_combined = LABELS_REAL_TARGET

# --- Dimensionality Reduction Functions ---
def reduce_dimensions(X: np.ndarray, method: str = 'pca', n_components: int = 3) -> Tuple[np.ndarray, Any]:
    print(f"Starting {method.upper()} reduction to {n_components} dimensions...")
    if method == 'tsne':
        reducer = TSNE(n_components=n_components, random_state=RANDOM_SEED, n_jobs=-1, init='pca', learning_rate='auto')
    elif method == 'pca':
        reducer = PCA(n_components=n_components, random_state=RANDOM_SEED)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")
        
    X_reduced = reducer.fit_transform(X)
    return X_reduced, reducer

# --- Goal 3: Visualize Distance (FULL IMPLEMENTATION with centering/scaling) ---
def visualize_proximity(X_high_dim: np.ndarray, word_labels: List[str], common_words: List[str], wordz_list: List[str]):
    """
    Visualizes the proximity between target words after centering and spreading them out.
    """
    print(f"\n--- Visualizing Target Embeddings Distance ---")  
    
    if X_high_dim.shape[0] == 0:
        print("🛑 Cannot visualize: No embeddings were successfully loaded.")
        return

    # 1. Center the Data (Subtract the mean)
    X_centered = X_high_dim - np.mean(X_high_dim, axis=0)

    # 2. Reduce Dimensionality using PCA
    X_3d_pca, _ = reduce_dimensions(X_centered, method='pca', n_components=3)

    # 3. Spread/Scale the Data (Ensure the plot space is filled)
    # Normalize max extent to ensure good separation visibility
    max_extent = np.max(np.abs(X_3d_pca))
    if max_extent > 0:
        X_3d_pca = X_3d_pca / max_extent * 10 # Scale to a visible range (e.g., -10 to 10)
    
    # 4. Create DataFrame
    df_pca = pd.DataFrame({
        'X': X_3d_pca[:, 0],
        'Y': X_3d_pca[:, 1],
        'Z': X_3d_pca[:, 2],
        'Word': word_labels,
    })
    
    # 5. Define 'Type' for coloring (Background is now implicitly removed)
    def get_type(word):
        if word in common_words:
            return 'English Concept'
        elif word in wordz_list:
            return 'aWORDZa Variant'
        else:
            return 'Other Target' # Fallback, should not happen if lists are correct
            
    df_pca['Type'] = df_pca['Word'].apply(get_type)
    df_plot = df_pca # Since we only pass targets, df_pca is df_plot

    # 6. Define Colors
    color_map = {
        'English Concept': 'steelblue', 
        'aWORDZa Variant': 'yellow', 
        'Other Target': 'red'
    }

    # 7. Generate Interactive 3D Scatter Plot
    fig = px.scatter_3d(
        df_plot,
        x='X',
        y='Y',
        z='Z',
        color='Type', 
        color_discrete_map=color_map,
        hover_name='Word',
        size=[15 for _ in df_plot['Type']], # Larger points for targets
        opacity=0.9,
        title=f'Goal 3: Proximity of Target Embeddings (Centered & Scaled)',
        height=700,
        template='plotly_dark'
    )
    
    # 8. Add Text Labels
    for index, row in df_plot.iterrows():
        text_color = 'steelblue' if row['Type'] == 'English Concept' else 'yellow'
        
        fig.add_trace(go.Scatter3d(
            x=[row['X']], y=[row['Y']], z=[row['Z']],
            mode='text',
            text=[row['Word']],
            textfont=dict(
                color=text_color, 
                size=12,
                family="Arial Black" if row['Type'] == 'aWORDZa Variant' else "Arial"
            ),
            textposition='top center',
            name=row['Word'],
            showlegend=False
        ))
        
    # 🛑 LINE DRAWING LOGIC HERE (Lines 0 to 0, 1 to 1, 2 to 2)
    if common_words and wordz_list:
        for eng, wordz in zip(common_words, wordz_list):
            try:
                # Retrieve coordinates for the English word and its Variant
                row_eng = df_plot[df_plot['Word'] == eng].iloc[0]
                row_wordz = df_plot[df_plot['Word'] == wordz].iloc[0]
                
                # Add the line trace
                fig.add_trace(go.Scatter3d(
                    x=[row_eng['X'], row_wordz['X']],
                    y=[row_eng['Y'], row_wordz['Y']],
                    z=[row_eng['Z'], row_wordz['Z']],
                    mode='lines',
                    # Using a thin, contrasting line for clarity
                    line=dict(color='rgba(255, 255, 0, 0.4)', width=2, dash='dot'),
                    showlegend=False
                ))
            except IndexError:
                # This should not happen with our setup, but is a good safeguard
                print(f"Skipping line: One or both of '{eng}', '{wordz}' not found in plot data.")
                pass

    fig.update_layout(
        paper_bgcolor='black',
        plot_bgcolor='black',
        scene=dict(
            xaxis=dict(title='PCA 1', backgroundcolor='black', gridcolor='gray', showbackground=False),
            yaxis=dict(title='PCA 2', backgroundcolor='black', gridcolor='gray', showbackground=False),
            zaxis=dict(title='PCA 3', backgroundcolor='black', gridcolor='gray', showbackground=False),
            # Ensure cube boundaries are centered around 0 and equal in scale
            aspectmode='cube',
            xaxis_range=[-12, 12],
            yaxis_range=[-12, 12],
            zaxis_range=[-12, 12]
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend_title_text='Word Type',
        font=dict(color='white')
    )
    
    fig.show()


# --- Final Execution ---
visualize_proximity(
    X_combined, 
    word_labels_combined, 
    COMMON_WORDS_REAL, 
    WORDZ_LIST_REAL
)