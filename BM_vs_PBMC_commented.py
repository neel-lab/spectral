import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
import warnings
import math

# Suppress runtime warnings (e.g., Seaborn log-scale or empty slice warnings) to keep console output clean
warnings.filterwarnings('ignore')

# ==============================================================================
# --- 1. Global Configuration ---
# ==============================================================================

# File system paths mapping to inputs (Parquet files) and desired output directory
BM_PATH = #path to folder containing BM files
PBMC_PATH = #path to folder containing PBMC files 
OUTPUT_DIR = #output path

# Color palettes for tissue source segregation (BMCs vs PBMCs)
PALETTE = {'BMCs': '#3498db', 'PBMCs': '#e74c3c'}         # Main high-contrast colors
PALETTE_LIGHT = {'BMCs': '#d1ecf1', 'PBMCs': '#f8d7da'}   # Translucent box fills

# Colormap for lectin intensity gradient plotting and data thresholding parameters
SENSITIVE_CMAP = 'rocket'
NOISE_FLOOR = 0.5   # Expression values below this floor are clamped in visualizations
ALPHA_STAT = 0.05   # Standard significance threshold for statistical testing

# Comprehensive list of glycan-binding lectins evaluated across the panels
LECTINS = ['UEA', 'CONA', 'PNA', 'GSLII', 'RCA', 'PHAE', 'MALII', 'GNA', 'WFA', 'LTL', 
           'SNA', 'DSL', 'GSL1B', 'HHL', 'SBA', 'PHAL', 'WGA', 'LEL', 'LCA', 'BanLec', 
           'AAL', 'ECL', 'VVA']

# Base antibody panel variables used to isolate immunophenotypic surface channels
PANEL_ANTIBODIES = [
    'CCR7', 'CD45RA', 'CD123', 'CD20', 'CD3', 'CD14', 'CD56', 'CD16', 
    'CD141', 'CD8', 'HLADR', 'CD25', 'CD4', 'IgD', 'TCRgd', 'CD11c', 
    'CD127', 'CD19', 'CD45', 'CD15', 'CD34', 'CD38'
]

# Normalization constants and downstream subsampling limits for visualization grids
COFACTOR = 100.0
TOP_N = 15           # Number of highly discriminative lectins extracted via linear SVM weights
CELLS_PER_PLOT = 8000  

# Dynamically generate required output directories if they do not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
for sub in ["Violin_Plots", "PCA_SVM_Grids", "Global_Stats"]:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)


# ==============================================================================
# --- 2. Unified Data Loading ---
# ==============================================================================

def load_and_preprocess():
    """
    Ingests raw Parquet datasets across tissue groups, unifies metadata nomenclature,
    filters relevant features, and applies inverse hyperbolic sine (arcsinh) scaling.
    """
    all_data_list = []
    tasks = [(BM_PATH, 'BMCs'), (PBMC_PATH, 'PBMCs')]
    
    for path, group_label in tasks:
        # Locate all individual Parquet data chunks inside target cohort directory
        files = list(Path(path).glob("*.parquet"))
        for f in files:
            df = pd.read_parquet(f)
            
            # Standardize cell label column formatting
            if 'Cell Type' in df.columns: 
                df.rename(columns={'Cell Type': 'Cell_Type'}, inplace=True)
            
            # Verify and cross-reference which target markers are present in this file instance
            available_lectins = [l for l in LECTINS if l in df.columns]
            available_abs = [col for col in df.columns if any(ab in col for ab in PANEL_ANTIBODIES)]
            
            # Subset table to relevant biological features and structural metadata columns
            df = df[available_lectins + available_abs + ['Cell_Type']].copy()
            numeric_cols = available_lectins + available_abs
            
            # Apply standard arcsinh normalization using the specified scaling cofactor matrix
            for col in numeric_cols: 
                df[col] = np.arcsinh(df[col] / COFACTOR)
            
            # Append experimental tracking features derived from context and filesystem values
            df['Group'] = group_label
            df['Sample'] = f.stem
            all_data_list.append(df)
            
    return pd.concat(all_data_list, ignore_index=True)

print("Loading Master Data...")
master_df = load_and_preprocess()

# --- NEW: Merge PBMC B Cell populations to match BMC nomenclature ---
# Standardizes developmental B-lineage terminology across the distinct sample sources
cell_type_mapping = {
    "Naive B": "Naive B MLZ",
    "Marginal Zone Like": "Naive B MLZ",
    "IgD- Memory B": "Memory B Plasmablasts",
    "Plasmablasts": "Memory B Plasmablasts"
}
master_df['Cell_Type'] = master_df['Cell_Type'].replace(cell_type_mapping)
# --------------------------------------------------------------------

# Establish lists of feature indices dynamically discovered inside the master table
lectin_cols = [l for l in LECTINS if l in master_df.columns]
found_abs = [col for col in master_df.columns if any(ab in col for ab in PANEL_ANTIBODIES)]

# Identify phenotypes that are common to both BMC and PBMC groups for matched profiling
shared_cells = sorted(list(set(master_df[master_df['Group'] == 'BMCs']['Cell_Type'].unique()).intersection(
    set(master_df[master_df['Group'] == 'PBMCs']['Cell_Type'].unique()))))

# Storage list to capture cross-validation classification accuracy scores per phenotype
performance_metrics = []


# ==============================================================================
# --- 3. Per Cell Type Modules ---
# ==============================================================================

for cell in shared_cells:
    print(f"Analyzing: {cell}")
    
    # Isolate records matching the current cell type and drop entries lacking glycan readouts
    type_df = master_df[master_df['Cell_Type'] == cell].dropna(subset=lectin_cols).copy()
    
    # Stratified downsampling to balance contributions evenly across donor biological replicates
    donors = type_df['Sample'].unique()
    cells_per_donor = CELLS_PER_PLOT // len(donors)
    subsampled = [type_df[type_df['Sample'] == d].sample(n=min(len(type_df[type_df['Sample'] == d]), cells_per_donor), random_state=42) for d in donors]
    plot_df = pd.concat(subsampled).copy()
    
    # --- PCA & SVM Logic ---
    # Extract complete multi-panel lectin matrices and build zero-indexed true source class vectors
    X_all = plot_df[lectin_cols].values
    y_corr = plot_df['Group'].map({'BMCs': 0, 'PBMCs': 1}).values
    
    # Run a complete PCA using all available lectin channels on the GPU or host environment
    pca_all = PCA(n_components=2).fit(X_all)
    X_pca_all = pca_all.transform(X_all)
    
    # Fit a linear Support Vector Classifier on the preliminary 2D projection space
    clf_all = SVC(kernel='linear', C=1.0, max_iter=50000).fit(X_pca_all, y_corr)
    
    # Dot product calculation mapping 2D decision boundary coefficients back to the initial high-dimensional lectin variables
    weights = np.dot(clf_all.coef_, pca_all.components_)[0]
    imp_df = pd.DataFrame({'Lectin': lectin_cols, 'Weight': weights})
    
    # Isolate top N highly discriminative markers based on the absolute magnitude of their combined directional weights
    top_lectins = imp_df.assign(Abs=imp_df['Weight'].abs()).sort_values('Abs', ascending=False).head(TOP_N)['Lectin'].tolist()
    top_lectins.sort()

    # Re-run specialized structural decomposition exclusively on the selected top discriminative lectin subset
    X_ref = plot_df[top_lectins].values
    pca_ref = PCA(n_components=2)
    X_pca_ref = pca_ref.fit_transform(X_ref)
    
    # Fit final production SVC classifier to evaluate localized multi-panel batch division limits
    clf_ref = SVC(kernel='linear', C=1.0, max_iter=50000).fit(X_pca_ref, y_corr)
    
    # Validate classification robustness using a 5-fold Stratified Cross-Validation protocol
    cv_acc = cross_val_score(clf_ref, X_pca_ref, y_corr, cv=StratifiedKFold(5, shuffle=True, random_state=42)).mean() * 100
    
    # Append the performance score to the summary tracker array
    performance_metrics.append(cv_acc)
    
    # Map coordinates back into the primary dataframe for plotting execution
    plot_df['PC1'], plot_df['PC2'] = X_pca_ref[:, 0], X_pca_ref[:, 1]

    # --- Shared Dynamic Group Label Helper ---
    # Computes tissue centroids within the low-dimensional map to place text indicators cleanly
    group_centroids = plot_df.groupby('Group')[['PC1', 'PC2']].mean()
    x_min_dyn, x_max_dyn = X_pca_ref[:, 0].min() - 1, X_pca_ref[:, 0].max() + 1
    y_min_dyn, y_max_dyn = X_pca_ref[:, 1].min() - 1, X_pca_ref[:, 1].max() + 1

    def apply_group_labels(target_ax, font_size=20):
        """Annotates tissue classification sectors using calculated spatial coordinates."""
        for g_name, pos in group_centroids.iterrows():
            tx = np.sign(pos['PC1']) * (abs(x_max_dyn if pos['PC1'] > 0 else x_min_dyn) * 0.75)
            ty = np.sign(pos['PC2']) * (abs(y_max_dyn if pos['PC2'] > 0 else y_min_dyn) * 0.75)
            target_ax.text(tx, ty, g_name, fontsize=font_size, fontweight='bold', color=PALETTE[g_name],
                           ha=('right' if tx > 0 else 'left'), va=('top' if ty > 0 else 'bottom'),
                           bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1))

    # --- Standard Meshgrid Configuration ---
    # Establishes coordinate grids to draw the underlying continuous SVM decision contours
    h = .1
    xx, yy = np.meshgrid(np.arange(X_pca_ref[:, 0].min()-2, X_pca_ref[:, 0].max()+2, h),
                         np.arange(X_pca_ref[:, 1].min()-2, X_pca_ref[:, 1].max()+2, h))
    Z = clf_ref.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # --------------------------------------------------------------------------
    # [A] Output PCA Lectin Grid
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(4, 4, figsize=(22, 20), sharex=True, sharey=True)
    fig.suptitle(f'{cell}: Lectin Grid (SVM Acc: {cv_acc:.1f}%)', fontsize=40, fontweight='bold', y=0.98)
    flat_axes = axes.flatten()
    
    for j in range(16):
        ax_curr = flat_axes[j]
        # Overlay the continuous hyper-plane decision background boundary
        ax_curr.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.1)
        
        if j == 0:
            # First panel displays categorical source scatter positions
            sns.scatterplot(x='PC1', y='PC2', hue='Group', data=plot_df, palette=PALETTE, s=15, ax=ax_curr, edgecolor=None, legend=False)
            ax_curr.set_title(f"Tissue Source\nAcc: {cv_acc:.1f}%", fontsize=18, fontweight='bold')
            apply_group_labels(ax_curr)
        elif j <= len(top_lectins):
            # Subsequent panels map single-marker glycan arcsinh intensity levels across coordinates
            lec = top_lectins[j-1]
            im = ax_curr.scatter(plot_df['PC1'], plot_df['PC2'], c=plot_df[lec], cmap=SENSITIVE_CMAP, vmin=NOISE_FLOOR, vmax=plot_df[lec].max(), s=10)
            fig.colorbar(im, ax=ax_curr, fraction=0.046, pad=0.04)
            ax_curr.set_title(lec, fontsize=18)
        else: 
            # Blank formatting for surplus grid axes
            ax_curr.axis('off')
            
    plt.tight_layout(rect=[0, 0, 1.0, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "PCA_SVM_Grids", f"{cell}_PCA_Lectin.png"), dpi=300); plt.close()

    # --------------------------------------------------------------------------
    # [B] Output Antibody PCA Grid
    # --------------------------------------------------------------------------
    current_abs = [ab for ab in found_abs if ab in plot_df.columns]
    if current_abs:
        rows_ab = math.ceil((len(current_abs) + 1) / 4)
        fig_ab, axes_ab = plt.subplots(rows_ab, 4, figsize=(22, 5 * rows_ab), sharex=True, sharey=True)
        fig_ab.suptitle(f'{cell}: Antibody Grid (SVM Acc: {cv_acc:.1f}%)', fontsize=40, fontweight='bold', y=0.98)
        flat_ab = axes_ab.flatten()
        
        for k in range(len(flat_ab)):
            ax_ab = flat_ab[k]
            # Overlay decision boundaries inside the antibody space
            ax_ab.contourf(xx, yy, Z, cmap='coolwarm', alpha=0.1)
            
            if k == 0:
                # Reference metadata plot
                sns.scatterplot(x='PC1', y='PC2', hue='Group', data=plot_df, palette=PALETTE, s=15, ax=ax_ab, edgecolor=None, legend=False)
                ax_ab.set_title(f"Source (Acc: {cv_acc:.1f}%)", fontsize=18, fontweight='bold')
                apply_group_labels(ax_ab)
            elif k <= len(current_abs):
                # Continuous color scatter mapping for surface phenotypic markers
                ab_col = current_abs[k-1]
                im_ab = ax_ab.scatter(plot_df['PC1'], plot_df['PC2'], c=plot_df[ab_col], cmap='viridis', s=10)
                plt.colorbar(im_ab, ax=ax_ab, fraction=0.046, pad=0.04)
                ax_ab.set_title(ab_col.split('_')[0], fontsize=18)
            else: 
                ax_ab.axis('off')
                
        plt.tight_layout(rect=[0, 0, 1.0, 0.95])
        plt.savefig(os.path.join(OUTPUT_DIR, "PCA_SVM_Grids", f"{cell}_PCA_Antibody.png"), dpi=300); plt.close()

    #

# --- 4. Final Summary ---
if performance_metrics:
    avg_perf = np.mean(performance_metrics)
    print(f"\n" + "="*40)
    print(f"GLOBAL AVERAGE SVM PERFORMANCE: {avg_perf:.2f}%")
    print(f"="*40)
else:
    print("No cell types were processed.")


# ==============================================================================
# --- 4. Module 3: Global Bubble Plot (Restored Formatting) ---
# ==============================================================================
print("Calculating Global Tissue-Specific Stats...")

# --- 1. Configuration (Bubble Script Workspace) ---
BMC_PATH = r"C:\Users\lebeatty\Box\cbe-neel-shared\JointProjects\Lauren\SingleCellGlycomics\SpectralFlow\SpecLec_data_analysis\Parquet Files\Bone Marrow\MDS\Healthy"
PBMC_PATH = r"C:\Users\lebeatty\Box\cbe-neel-shared\JointProjects\Lauren\SingleCellGlycomics\Parquet Files\PBMC"
OUTPUT_PATH = r"C:\Users\lebeatty\Box\cbe-neel-shared\JointProjects\Lauren\SingleCellGlycomics\MDS_Study"

COFACTOR = 100.0  
ALPHA = 0.05       # Significance threshold (FDR p-adj boundary)

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Categorized grouping of lectins by standard mammalian carbohydrate structures
specificity_groups = {
    "GlcNAc": ['GSLII', 'WGA'],
    "LacNAc": ['LEL', 'DSL', 'ECL', 'RCA'],
    "Gal": ['GSL1B', 'PNA'],
    "GalNAc": ['WFA', 'SBA', 'VVA'],
    "Sia": ['SNA', 'MALII'],
    "Mannose": ['CONA', 'GNA', 'HHL', 'BanLec'],
    "Branched": ['PHAE', 'PHAL'],
    "Fucose": ['UEA', 'AAL', 'LTL', 'LCA']
}

# --- 2. Data Loading & Arcsinh Normalization ---
dfs = []
def load_dir(path, label):
    """Imports Parquet data chunks and creates structural metadata tracking variables."""
    files = sorted([f for f in os.listdir(path) if f.endswith('.parquet')])
    for filename in files:
        df = pd.read_parquet(os.path.join(path, filename))
        if "Cell Type" in df.columns: 
            df = df.rename(columns={"Cell Type": "Cell_Type"})
        df["Replicate"] = f"{label}_{filename.replace('.parquet', '')}"
        df["Sample_Type"] = label
        dfs.append(df)

print("Loading data...")
load_dir(BMC_PATH, "BMCs")
load_dir(PBMC_PATH, "PBMCs")
full = pd.concat(dfs, ignore_index=True)

# --- NEW: Merge PBMC B Cell populations to match BMC nomenclature ---
full['Cell_Type'] = full['Cell_Type'].replace(cell_type_mapping)
# --------------------------------------------------------------------

# Construct targeted list of discovered lectins mapped inside the input data tables
lectins_in_data = [l for group in specificity_groups.values() for l in group if l in full.columns]

# --- 2.5 Filter for Common Cell Types ---
# Interrogate cell classifications across the groups to isolate clean intersections
bmc_types = set(full[full["Sample_Type"] == "BMCs"]["Cell_Type"].unique())
pbmc_types = set(full[full["Sample_Type"] == "PBMCs"]["Cell_Type"].unique())

print(f"\n--- Unique Cell Types in BMCs ({len(bmc_types)}) ---")
for ct in sorted(bmc_types): print(f"  - {ct}")

print(f"\n--- Unique Cell Types in PBMCs ({len(pbmc_types)}) ---")
for ct in sorted(pbmc_types): print(f"  - {ct}")

# Extract phenotype intersection list
common_types = list(bmc_types.intersection(pbmc_types))
print(f"\n---> Common cell types being analyzed: {len(common_types)}")

# Filter primary tables down to matched shared cellular subsets
full = full[full["Cell_Type"].isin(common_types)].reset_index(drop=True)

# Loop to execute per-replicate arcsinh normalization transformations safely
processed_list = []
for rep, grp in full.groupby("Replicate"):
    grp = grp.copy()
    grp[lectins_in_data] = np.arcsinh(grp[lectins_in_data].values / COFACTOR)
    processed_list.append(grp)

all_transformed = pd.concat(processed_list, ignore_index=True)


# ==============================================================================
# --- 3. Statistical Testing: BMCs vs PBMCs per Cell Type ---
# ==============================================================================

# Aggregate single-cell arrays to Pseudobulk space (deriving median per donor replicate per phenotype)
pbulk = all_transformed.groupby(["Replicate", "Cell_Type", "Sample_Type"])[lectins_in_data].median().reset_index()
records = []
cell_types = pbulk["Cell_Type"].unique()

for lectin in lectins_in_data:
    for ct in cell_types:
        # Isolate pseudobulk cluster targets matching current specifications
        ct_data = pbulk[pbulk["Cell_Type"] == ct]
        
        mds_vals = ct_data.loc[ct_data["Sample_Type"] == "BMCs", lectin].dropna().values
        healthy_vals = ct_data.loc[ct_data["Sample_Type"] == "PBMCs", lectin].dropna().values
        
        # Welch's T-test calculation (robust against unequal sample distribution shapes and variances)
        if len(mds_vals) > 1 and len(healthy_vals) > 1:
            stat, pval = ttest_ind(mds_vals, healthy_vals, equal_var=False, nan_policy='omit')
            val_diff = np.nanmedian(mds_vals) - np.nanmedian(healthy_vals)
        else:
            stat, pval, val_diff = np.nan, np.nan, np.nan
        
        records.append({
            "Lectin": lectin, 
            "Cell_Type": ct, 
            "t_stat": stat, 
            "pval": pval, 
            "Intensity": val_diff
        })

ttest_df = pd.DataFrame.from_records(records)

# Apply Benjamini-Hochberg False Discovery Rate (FDR) correction over valid numeric tests
mask = ttest_df['pval'].notna()
if mask.any():
    ttest_df.loc[mask, 'p_adj'] = multipletests(ttest_df.loc[mask, 'pval'], method='fdr_bh')[1]
else:
    ttest_df['p_adj'] = np.nan

# Construct localized index sheets for spatial circle plot mapping execution
pivot_diff = ttest_df.pivot(index='Cell_Type', columns='Lectin', values='Intensity')
pvalue_df = ttest_df.copy()


# ==============================================================================
# --- 4. Hierarchical Clustering (Rows Only, NaN-Safe) ---
# ==============================================================================
clean_pivot = pivot_diff.dropna() 
if not clean_pivot.empty:
    # Run agglomerative hierarchical linkage optimization using Ward's minimal variance criteria
    linkage_matrix = linkage(clean_pivot.values, method='ward', metric='euclidean')
    dendro_info = dendrogram(linkage_matrix, labels=clean_pivot.index.tolist(), no_plot=True)
    
    # Establish ordered row index arrays based on dendrogram leaf sorting optimization
    ordered_cell_types = [clean_pivot.index[i] for i in dendro_info['leaves']]
    
    # Append cells that contained missing entries back onto the outer margins of the index
    remaining = [c for c in pivot_diff.index if c not in ordered_cell_types]
    ordered_cell_types.extend(remaining)
    pivot_diff = pivot_diff.reindex(ordered_cell_types)
else:
    ordered_cell_types = pivot_diff.index.tolist()


# ==============================================================================
# --- 5. Lectin Grouping & Formatting for Plot ---
# ==============================================================================
group_palette = plt.get_cmap('tab10')
group_colors = {group: group_palette(i) for i, group in enumerate(specificity_groups.keys())}

# Establish explicit color matching dicts to sync text labels with target glycan groups
lectin_color_map = {}
for group, lectins in specificity_groups.items():
    c = group_colors[group]
    for lectin in lectins:
        lectin_color_map[lectin] = c

# Re-order and document bounding track indices for block structure category lines
ordered_plot_lectins = []
group_ranges = [] 
current_idx = 0
for group_name, lec_list in specificity_groups.items():
    valid_lecs = [l for l in lec_list if l in lectins_in_data]
    if valid_lecs:
        start = current_idx
        ordered_plot_lectins.extend(valid_lecs)
        end = current_idx + len(valid_lecs) - 1
        group_ranges.append((group_name, start, end))
        current_idx += len(valid_lecs)


# ==============================================================================
# --- 6. Circle Plot Construction ---
# ==============================================================================
plt.rcParams['font.family'] = 'Arial'
fig = plt.figure(figsize=(35, 45))
gs = fig.add_gridspec(1, 3, width_ratios=[10, 0.8, 0.3], wspace=0.05)
ax = fig.add_subplot(gs[0])

# Configure symmetric color boundaries for diverging heatmap scale
limit = max(abs(pivot_diff.values.min()), abs(pivot_diff.values.max()))
vmin, vmax = -limit, limit
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# Construct custom color transition gradient (Blue -> White -> Orange-Red)
colors = ["#0000ff", "#ffffff", "#ff4500"] 
vanimo_white = mcolors.LinearSegmentedColormap.from_list("vanimo_white", colors)
cmap = vanimo_white

# Bounding limits mapping raw p-values linearly to circle element radii
p_min_data = 0.02
p_max_data = 0.98
min_radius = 0.08
max_radius = 0.4 

# Font dimensions scaling parameter config
title_size, axis_label_size, group_label_size, colorbar_label_size = 85, 50, 45, 50

# Main loop generating circle geometries across the phenotype vs lectin grid
for i, ct in enumerate(pivot_diff.index):
    for j, lectin in enumerate(ordered_plot_lectins):
        if lectin not in pivot_diff.columns: continue
            
        row = pvalue_df[(pvalue_df["Cell_Type"] == ct) & (pvalue_df["Lectin"] == lectin)]
        val = pivot_diff.loc[ct, lectin]
        
        if pd.isna(val) or row.empty or pd.isna(row["p_adj"].values[0]):
            continue
            
        padj = float(row["p_adj"].values[0])
        
        # Run linear parameter interpolation to map inverse p-value ranges to marker sizing radii
        p_min_data = ttest_df['p_adj'].min()
        p_max_data = ttest_df['p_adj'].max()
        
        p_range = p_max_data - p_min_data if p_max_data != p_min_data else 1.0
        norm_p = (padj - p_min_data) / p_range
        radius_factor = 1 - norm_p 
        radius = min_radius + (radius_factor * (max_radius - min_radius))
        
        # Highlight statistically significant hits with a pronounced dark border outline
        color = cmap(norm(val))
        lw = 5 if padj <= ALPHA else 1
        ec = "black" if padj <= ALPHA else "darkgray"
        
        circ = plt.Circle((j, i), radius, facecolor=color, edgecolor=ec, lw=lw)
        ax.add_patch(circ)

# Axis ticks and orientation styling labels setup
ax.set_xticks(range(len(ordered_plot_lectins)))
ax.set_xticklabels(ordered_plot_lectins, rotation=90, fontsize=axis_label_size, fontweight='bold')
ax.set_yticks(range(len(pivot_diff.index)))
ax.set_yticklabels(pivot_diff.index, fontsize=axis_label_size, fontweight='bold')

# Apply color values to single text labels matching active carbohydrate groupings
for label in ax.get_xticklabels():
    lectin_name = label.get_text()
    if lectin_name in lectin_color_map:
        label.set_color(lectin_color_map[lectin_name])

ax.set_xlim(-0.5, len(ordered_plot_lectins)-0.5)
ax.set_ylim(-0.5, len(pivot_diff.index)-0.5)
ax.set_title("Lectin Binding: BMCs vs PBMCs", fontsize=title_size, fontweight='bold', pad=40)

# Generate bounding underline indicators for global carbohydrate structural classes
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
for group, start, end in group_ranges:
    c = group_colors[group]
    ax.add_line(Line2D([start, end], [-0.08, -0.08], transform=trans, color=c, linewidth=5, clip_on=False))
    ax.text((start + end) / 2, -0.10, group, transform=trans, 
            ha='center', va='top', fontsize=group_label_size, fontweight='bold', color=c) 

# Add structural dendrogram profile graph on side subplot panel
if not clean_pivot.empty:
    ax_dendro = fig.add_subplot(gs[1])
    dendrogram(linkage_matrix, orientation='right', no_labels=True, color_threshold=0, above_threshold_color='black')
    ax_dendro.axis('off')

# Render master colorbar mapping coordinate definitions
ax_cbar = fig.add_subplot(gs[2])
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar = plt.colorbar(sm, cax=ax_cbar)
cbar.set_label("$\Delta$ Median Binding (Arcsinh Difference)", fontsize=colorbar_label_size, fontweight='bold')
cbar.ax.tick_params(labelsize=colorbar_label_size)

plt.subplots_adjust(left=0.1, right=0.85, top=0.95, bottom=0.20)
plt.show()


# ==============================================================================
# --- 7. Univariate Profile Boxplots per Cell Type ---
# ==============================================================================
print("Generating Univariate Profile Boxplots...")

# Define specialized file folder endpoints for distribution violin/boxplot charting
boxplot_dir = r'C:\Users\lebeatty\Box\cbe-neel-shared\JointProjects\Lauren\SingleCellGlycomics\SpectralFlow\Bone Marrow Panel\violin_plots'
os.makedirs(boxplot_dir, exist_ok=True)

# Synchronize discrete donor fill definitions matching continuous pipeline maps
PALETTE_LIGHT_MAP = {'BMCs': '#d1ecf1', 'PBMCs': '#f8d7da'}
PALETTE_MAP = {'BMCs': '#3498db', 'PBMCs': '#e74c3c'}

# State parameter ensuring legend assets only output once across loop execution paths
legend_saved = False 

for ct in common_types:
    ct_data = pbulk[pbulk["Cell_Type"] == ct]
    if ct_data.empty: 
        continue

    # Pivot table arrays from wide form to narrow long formats required for Seaborn functions
    melted_df = ct_data.melt(
        id_vars=["Replicate", "Sample_Type"],
        value_vars=[l for l in ordered_plot_lectins if l in pbulk.columns],
        var_name="Lectin",
        value_name="Intensity"
    )

    plt.figure(figsize=(20, 8))

    # [Step 1]: Draw baseline distribution boxes using muted translucent filling and gray margins
    ax = sns.boxplot(
        x="Lectin", y="Intensity", hue="Sample_Type", data=melted_df,
        palette=PALETTE_LIGHT_MAP, showfliers=False, width=0.6,
        boxprops={'edgecolor': 'darkgray', 'zorder': 1},
        medianprops={'color': 'darkgray', 'zorder': 1},
        whiskerprops={'color': 'darkgray', 'zorder': 1},
        capprops={'color': 'darkgray', 'zorder': 1}
    )

    # [Step 2]: Overlay fine strip dots representing separate biological sample replicate donors
    sns.stripplot(
        x="Lectin", y="Intensity", hue="Sample_Type", data=melted_df,
        palette=PALETTE_MAP, dodge=True, size=6, alpha=0.9,
        edgecolor='white', linewidth=0.5, zorder=2, ax=ax
    )

    # Graph text labels configuration formatting lines
    plt.title(f"{ct}: BMCs vs PBMCs", fontsize=32, fontweight='bold', pad=15)
    plt.xlabel("Lectin", fontsize=26, fontweight='bold', labelpad=15)
    plt.ylabel("Intensity", fontsize=22, fontweight='bold', labelpad=15)
    plt.xticks(rotation=45, ha='right', fontsize=20)
    plt.yticks(fontsize=20) 

    # Remove duplicate trace entries generated by combinations of box and strip plots
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)
     
    # --- STANDALONE LEGEND EXPORT BRANCH ---
    # Generates a distinct high-resolution standalone legend file instance on first loop pass
    if not legend_saved and unique_handles:
        fig_leg, ax_leg = plt.subplots(figsize=(6, 4))
        ax_leg.axis('off') 
         
        ax_leg.legend(
            unique_handles, 
            unique_labels, 
            title="Group", 
            loc='center', 
            framealpha=1, 
            fontsize=22,         
            title_fontsize=26,   
            markerscale=2        
        )
         
        fig_leg.savefig(os.path.join(boxplot_dir, "Legend_Standalone.png"), dpi=300, bbox_inches='tight')
        plt.close(fig_leg)
        legend_saved = True

    # Cleanly strip out legend assets from main layout panels to preserve alignment
    if ax.get_legend() is not None:
        ax.get_legend().remove()

    # Commit output figures safely back down to disk storage paths
    plt.tight_layout()
    safe_ct_name = ct.replace("/", "_").replace(" ", "_")
    plt.savefig(os.path.join(boxplot_dir, f"{safe_ct_name}_Univariate.png"), dpi=300)
    plt.close()

print(f"All Univariate Boxplots and standalone legend saved to: {boxplot_dir}")
