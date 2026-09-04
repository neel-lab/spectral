import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import harmonypy as hm
import torch
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# 0. CONFIGURATION & MODE ASSIGNMENT
# ==========================================

# Select the panel operational mode here: Choose either '3 panel' or '5 panel'
PANEL_MODE = '3 panel' 

if PANEL_MODE == '3 panel':
    # Define panel list and specific lectins used in the 3-panel experimental configuration
    panel_folders = ['Panel 1', 'Panel 2', 'Panel 3']
    p1_lectins = ['PHAL', 'CONA', 'GNA', 'GSLII', 'RCA']
    p2_lectins = ['PHAE', 'DSL', 'ECL', 'WFA', 'LTL']
    p3_lectins = ['SNA', 'WGA', 'VVA', 'LCA', 'AAL']
    all_lectins = p1_lectins + p2_lectins + p3_lectins

elif PANEL_MODE == '5 panel':
    # Define panel list and specific lectins used in the 5-panel experimental configuration
    panel_folders = ['Panel 1', 'Panel 2', 'Panel 3', 'Panel 4', 'Panel 5']
    p1_lectins = ['UEA', 'CONA', 'PNA', 'GSLII', 'RCA']
    p2_lectins = ['PHAE', 'MALII', 'GNA', 'WFA', 'LTL']
    p3_lectins = ['SNA', 'DSL', 'GSL1B', 'HHL', 'SBA']
    p4_lectins = ['PHAL', 'WGA', 'LEL', 'LCA']
    p5_lectins = ['BanLec', 'AAL', 'ECL', 'VVA']
    all_lectins = p1_lectins + p2_lectins + p3_lectins + p4_lectins + p5_lectins
else:
    raise ValueError("Invalid PANEL_MODE. Choose '3 panel' or '5 panel'.")


# ==========================================
# 1. PARAMETER CONFIGURATION & RUNTIME SETUP
# ==========================================

# Configure PyTorch CUDA cache allocators to reduce virtual memory fragmentation during heavy KNN sweeps
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")

# Define target workspace path arrays containing absolute location maps for sample replicates
root_folders = [
  
]

# Non-biological parameters, scatter profiles, and autofluorescence channels excluded from analysis
columns_to_drop = ['Time', 'SSC-H', 'SSC-A', 'FSC-H', 'FSC-A', 'SSC-B-H', 'SSC-B-A', 'AF-A', 'CF700-A', 'BUV395-A']

# Subsampling cap parameters to keep dataset parsing boundaries lightweight
subsample_n = 2000000
k = 3

# Compute backend hardware selection (GPU acceleration via CUDA if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ==========================================
# 2. COMPUTATIONAL CORE FUNCTIONS (GPU)
# ==========================================

def knn_torch_streaming(
    train_embeddings,          # numpy array or torch tensor (n_train, dim)
    train_values,              # numpy array or torch tensor (n_train,)
    query_embeddings,         # numpy array or torch tensor (n_query, dim)
    k,
    query_batch_size=1024,     # tune down if OOM (smaller -> less memory)
    index_chunk_size=32768,    # tune down if OOM (smaller -> less memory)
    use_fp16=True,             # convert embeddings to float16 on GPU to reduce memory
    verbose=False
):
    """
    Memory-efficient KNN by streaming the index in chunks.
    Returns: numpy array of shape (n_query,) with mean of neighbor train_values for each query.
    """
    # Cast input arrays to native torch tensors
    if not torch.is_tensor(train_embeddings):
        train_embeddings = torch.from_numpy(train_embeddings)
    if not torch.is_tensor(train_values):
        train_values = torch.from_numpy(train_values)
    if not torch.is_tensor(query_embeddings):
        query_embeddings = torch.from_numpy(query_embeddings)

    # Migrate working tensors onto the selected hardware device runtime target
    train_embeddings = train_embeddings.to(device)
    train_values = train_values.to(device)
    query_embeddings = query_embeddings.to(device)

    # Apply half-precision FP16 operations if toggled to reduce global VRAM usage by 50%
    if use_fp16:
        train_embeddings = train_embeddings.half()
        query_embeddings = query_embeddings.half()
    else:
        train_embeddings = train_embeddings.float()
        query_embeddings = query_embeddings.float()

    n_train = train_embeddings.shape[0]
    n_query = query_embeddings.shape[0]
    dim = train_embeddings.shape[1]
    k_use = min(k, n_train)
    
    if verbose:
        print(f"[knn_stream] n_train={n_train}, n_query={n_query}, dim={dim}, k_use={k_use}")
        
    # Initialize CPU-bound storage matrix for consolidated outputs to protect active GPU memory limits
    results = torch.empty(n_query, device="cpu", dtype=train_values.dtype)

    # Isolate memory blocks and clear execution context backends from tracking gradients
    torch.cuda.empty_cache()
    with torch.no_grad():
        # Outer streaming loop: Process specific batches of cells requiring imputation
        for qstart in range(0, n_query, query_batch_size):
            qend = min(qstart + query_batch_size, n_query)
            q_batch = query_embeddings[qstart:qend]
            bq = q_batch.shape[0]

            # Construct empty placeholder matrices to retain rolling top-K closest indexes
            D_top = torch.full((bq, k_use), float("inf"), device=device, dtype=torch.float32)
            I_top = torch.full((bq, k_use), -1, device=device, dtype=torch.long)

            # Inner streaming loop: Iteratively sweep query blocks against slices of reference training data
            for tstart in range(0, n_train, index_chunk_size):
                tend = min(tstart + index_chunk_size, n_train)
                idx_chunk = train_embeddings[tstart:tend]  
                chunk_n = idx_chunk.shape[0]

                # Compute pairwise Euclidean distances across the selected spatial configurations
                if use_fp16:
                    distances = torch.cdist(q_batch.float(), idx_chunk.float(), p=2)  
                else:
                    distances = torch.cdist(q_batch, idx_chunk, p=2)  

                # Extract local minimum neighbors inside the current chunk bounds
                k_chunk = min(k_use, chunk_n)
                dk, ik = torch.topk(distances, k_chunk, largest=False, dim=1)  

                # Re-index local matrix positions into absolute training coordinates
                ik_global = ik + tstart  

                # Merge spatial distances from current block with previous global minimum values
                D_cat = torch.cat([D_top, dk.to(dtype=torch.float32)], dim=1)   
                I_cat = torch.cat([I_top, ik_global], dim=1)                     

                # Resolve consolidated top-K neighbors across concatenated references
                D_top, idx_in_cat = torch.topk(D_cat, k_use, largest=False, dim=1)  
                I_top = torch.gather(I_cat, 1, idx_in_cat)

                # Clear working memory matrices within the current index chunk block to prevent VRAM accumulation
                del distances, dk, ik, ik_global, D_cat, I_cat, idx_in_cat
                torch.cuda.empty_cache()

            # Extrapolate imputed target marker concentrations by computing mean values across top-K neighbors
            neighbor_vals = train_values[I_top]  
            filled_batch = neighbor_vals.float().mean(dim=1)  

            # Move batch calculations back into host CPU memory map structures
            results[qstart:qend] = filled_batch.cpu()

            if verbose:
                print(f"[knn_stream] processed queries {qstart}:{qend}")

            # Clear out lingering query loop intermediate variables
            del D_top, I_top, neighbor_vals, filled_batch
            torch.cuda.empty_cache()

    return results.numpy()


def gpu_pca(X_np, n_components=15):
    """
    Perform PCA on GPU using PyTorch
    """
    # Move to GPU and convert to float32
    X_t = torch.from_numpy(X_np.astype('float32')).to(device)
    
    # Standardize data to zero-mean across features
    X_mean = X_t.mean(dim=0, keepdim=True)
    Xc = X_t - X_mean
    
    # Compute low-rank Singular Value Decomposition (SVD) for optimal GPU resource utilization
    U, S, V = torch.pca_lowrank(Xc, q=n_components, center=False)
    
    # Derivatively calculate the final PC factor score matrix projections
    pca_scores = U * S.unsqueeze(0)  
    
    return pca_scores.cpu().numpy()


# ==========================================
# 3. SAMPLE DATA INGESTION & COHORT PROCESSING
# ==========================================

# Iterate processing loops across individual donor storage directories
for root_folder in root_folders:
    print(f"\n🔄 Processing folder: {root_folder}")
    
    cleaned_dfs = {}
    cleaned_dfs_unnormalized = {}  
    
    # Aggregate data files from panel specific folders
    for panel in panel_folders:
        folder_path = os.path.join(root_folder, panel)
        df_list = []

        # Interrogate files in the target directory and import single cell expression measurements
        for filename in os.listdir(folder_path):
            if filename.endswith(".csv") and not filename.startswith("combined_"):
                df = pd.read_csv(os.path.join(folder_path, filename))
                df['Cell Type'] = os.path.splitext(filename)[0] # Extract phenotypical marker from file name
                df['Sample'] = panel # Store originating batch designator metadata
                df_list.append(df)

        # Concatenate experimental tables into unified dataframe matrices
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df.columns = combined_df.columns.str.strip()
        combined_df = combined_df.drop(columns=[c for c in columns_to_drop if c in combined_df.columns])

        # Preserve a dedicated duplicate copy containing pristine unscaled raw baseline profiles
        combined_df_unnormalized = combined_df.copy()
        
        # Standardize anchor protein expression structures using Z-score transformations
        ab_cols = [c for c in combined_df.columns if c not in all_lectins + ['Sample', 'Cell Type'] and not c.startswith("Unnamed")]
        scaler = StandardScaler()
        combined_df[ab_cols] = scaler.fit_transform(combined_df[ab_cols])

        # Store working tables within dynamic evaluation registry arrays
        cleaned_dfs[f"combined_{panel.replace(' ', '').lower()}"] = combined_df
        cleaned_dfs_unnormalized[f"combined_{panel.replace(' ', '').lower()}"] = combined_df_unnormalized

    # Map pipeline handles dynamically based on the active panel mode selection
    all_panels = [cleaned_dfs[f"combined_{p.replace(' ', '').lower()}"] for p in panel_folders]
    all_panels_unnorm = [cleaned_dfs_unnormalized[f"combined_{p.replace(' ', '').lower()}"] for p in panel_folders]
    
    # Apply subsampling constraints while matching exact cellular row indices across both sets
    sampled_indices = [df.sample(n=min(subsample_n, len(df)), random_state=42).index for df in all_panels]
    
    # Flatten across panel layouts to establish unified cell coordinate systems
    full_df = pd.concat([all_panels[i].loc[sampled_indices[i]] for i in range(len(all_panels))], ignore_index=True)
    full_df_unnorm = pd.concat([all_panels_unnorm[i].loc[sampled_indices[i]] for i in range(len(all_panels_unnorm))], ignore_index=True)

    # Drop cell entries missing baseline anchor features and synchronize across variants
    Ab_cols = [c for c in full_df.columns if c not in all_lectins + ['Cell Type', 'Sample']]
    full_df = full_df.dropna(subset=Ab_cols)
    full_df_unnorm = full_df_unnorm.loc[full_df.index]  
    
    X_full_np = full_df[Ab_cols].values
    
    # Extract structural components on shared features using GPU-accelerated PCA
    pca_result_full = gpu_pca(X_full_np, n_components=15)
    
    # Stabilize structural technical variations and plate effects via Harmony integration
    ho_full = hm.run_harmony(pca_result_full, full_df, vars_use=['Sample'], max_iter_harmony=20)
    harmony_embeddings = pd.DataFrame(ho_full.Z_corr, index=full_df.index)

    # Initialize destination object using pristine unnormalized matrix values
    filled_df = full_df_unnorm.copy()


    # ==========================================
    # 4. CROSS-PANEL MARKER EVALUATION & IMPUTATION
    # ==========================================
    
    # Organize lectin loops dynamically using the active configuration mappings
    if PANEL_MODE == '3 panel':
        panel_lectins_list = [p1_lectins, p2_lectins, p3_lectins]
    else:
        panel_lectins_list = [p1_lectins, p2_lectins, p3_lectins, p4_lectins, p5_lectins]

    for panel_lectins in panel_lectins_list:
        for lectin in panel_lectins:
            # Isolate cellular indices that contain active physical readings for training
            train_mask = ~full_df_unnorm[lectin].isna()
            if train_mask.sum() == 0:
                continue

            # Load known reference points into low-dimensional numpy arrays
            X_train_np = harmony_embeddings.loc[train_mask].values.astype('float32')
            y_train_np = full_df_unnorm.loc[train_mask, lectin].values.astype('float32')  
            
            # --- COMPUTE RMSE VALIDATION METRIC ---
            # Randomly hold back 20% of the measured cells to validate imputation accuracy
            if len(y_train_np) >= 10:  
                np.random.seed(42)
                shuffled_indices = np.random.permutation(len(y_train_np))
                val_size = int(len(y_train_np) * 0.20)
                
                val_idx = shuffled_indices[:val_size]
                train_idx = shuffled_indices[val_size:]
                
                if len(val_idx) > 0 and len(train_idx) > 0:
                    # Impute expression on the validation test cells using the remaining 80% reference set
                    val_pred = knn_torch_streaming(
                        train_embeddings=X_train_np[train_idx],
                        train_values=y_train_np[train_idx],
                        query_embeddings=X_train_np[val_idx],
                        k=k,
                        query_batch_size=512,
                        index_chunk_size=16384,
                        use_fp16=True,
                        verbose=False
                    )
                    # Evaluate success by measuring RMSE and R2 scores against known ground truths
                    rmse_score = np.sqrt(mean_squared_error(y_train_np[val_idx], val_pred))
                    r2_score_val = r2_score(y_train_np[val_idx], val_pred)
                    print(f"📈 Evaluation -> Lectin: {lectin:<7} | RMSE: {rmse_score:.4f} | R2: {r2_score_val:.4f}")

            # Identify target destinations containing missing values requiring feature-space imputation
            query_mask = full_df_unnorm[lectin].isna()
            if query_mask.sum() == 0:
                continue

            # Extract Harmony spatial map layouts for query items
            X_query_np = harmony_embeddings.loc[query_mask].values.astype('float32')
            
            # Execute distance-weighted streaming matrix imputation across missing gaps
            filled_values = knn_torch_streaming(
                train_embeddings=X_train_np,
                train_values=y_train_np,
                query_embeddings=X_query_np,
                k=3,
                query_batch_size=512,
                index_chunk_size=16384,
                use_fp16=True,
                verbose=True
            )
            
            # Write imputed values back into the unified dataframe structure
            filled_df.loc[query_mask, lectin] = filled_values

    # Integrate the low-dimensional Harmony alignment axes directly into output structures
    harmony_cols = [f'Harmony_{i+1}' for i in range(harmony_embeddings.shape[1])]
    for i, col in enumerate(harmony_cols):
        filled_df[col] = harmony_embeddings.iloc[:, i].values

    # Round continuous variables to 2 decimal places to clean up signal float artifacts
    lectin_columns = [col for col in filled_df.columns if col in all_lectins]
    filled_df[lectin_columns] = filled_df[lectin_columns].round(2)
    
    # Export fully integrated and imputed single-cell dataset matrices back to disk
    filled_df.to_csv(os.path.join(root_folder, "filled_harmony_knn.csv"), index=False)
    print(f"✅ Saved: filled_harmony_knn_full.csv for {root_folder}")