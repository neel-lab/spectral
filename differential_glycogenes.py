import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import random

# =============================================================================
# 0. GLOBAL PLOT CONFIGURATION & TYPOGRAPHY
# =============================================================================
# Configure universal Matplotlib rendering engine defaults to enforce high-contrast,
# publication-ready typography across all subplots, annotations, and legends.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'

# =============================================================================
# 1. FILE PATH CONFIGURATION
# =============================================================================
# Absolute path allocations for raw transcriptomic differential expression data 
# and structural glycogenomic annotations.
path_de = r"pathto\differential_expression_results.csv" #file output from CellXgene differential expression
path_glyco = r"pathto\glycogenes.csv" #list of glycogenes to narrow search to

# =============================================================================
# 2. DATA ACQUISITION, STRUCTURING, & CLEANING
# =============================================================================
# Load tabular datasets while explicitly omitting parsing commentary metrics
de_results = pd.read_csv(path_de, comment='#')
glyco_df = pd.read_csv(path_glyco)

# Coerce expression log values to numeric float types and drop incomplete records
de_results['Log Fold Change'] = pd.to_numeric(de_results['Log Fold Change'], errors='coerce')
de_results = de_results.dropna(subset=['Log Fold Change'])

# Mathematical inversion of the expression ratio directionality matrix:
# Forces Positive values = Bone Marrow up-regulation; Negative values = Blood up-regulation
de_results['Log Fold Change'] = de_results['Log Fold Change'] * -1 

# Standardize nomenclature datasets via string stripping and uppercase alignment
de_results['Gene'] = de_results['Gene'].astype(str).str.strip().str.upper()
glyco_df['Gene'] = glyco_df['Gene'].astype(str).str.strip().str.upper()
glyco_df['Class'] = glyco_df['Class'].astype(str).str.strip()
glyco_df['Pathway'] = glyco_df['Pathway'].astype(str).str.strip()

# =============================================================================
# 3. TRANSCRIPTOME FILTERING & TAXONOMY CORRELATION
# =============================================================================
# Isolate global expression profiles down strictly to confirmed glycogenomic targets
filtered_de = de_results[de_results['Gene'].isin(glyco_df['Gene'])].copy()
filtered_de = filtered_de.merge(glyco_df[['Gene', 'Class', 'Pathway']], on='Gene', how='left')

# Scrub missing categories and filter out non-informative string designations
filtered_de = filtered_de.dropna(subset=['Class', 'Pathway'])
filtered_de = filtered_de[~filtered_de['Class'].str.lower().isin(['nan', 'none'])]

# Calculate vector absolute magnitude to isolate the highest-variance effect sizes
filtered_de['Magnitude'] = filtered_de['Log Fold Change'].abs()
top_50 = filtered_de.nlargest(50, 'Magnitude').sort_values('Log Fold Change', ascending=False)

# =============================================================================
# 4. CATEGORICAL COLOR-MAPPING ENGINE
# =============================================================================
def get_map_and_palette(df, column, is_pathway=False):
    """
    Generates discrete categorical index maps and custom qualitative color palettes 
    for multi-axial categorical tracking sidebars.
    """
    unique_vals = sorted(df[column].unique())
    n_colors = len(unique_vals)
    val_map = {val: i for i, val in enumerate(unique_vals)}
    
    if is_pathway:
        # Complex multi-categorical blend for wide-spectrum metabolic pathways
        p1, p2, p3 = sns.color_palette("tab20"), sns.color_palette("tab20b"), sns.color_palette("tab20c")
        combined_pal = p1 + p2 + p3
        random.seed(42)  # Enforce reproducibility of the categorical shuffle
        random.shuffle(combined_pal)
        palette = combined_pal[:n_colors] if n_colors <= len(combined_pal) else sns.color_palette("husl", n_colors)
    else:
        # Standard compact categorical tracking for high-level functional groups
        palette = sns.color_palette("tab10", n_colors) if n_colors <= 10 else sns.color_palette("tab20", n_colors)
        
    return val_map, palette, unique_vals

# Build lookup coordinate matrices and color arrays for the dual sidebars
class_map, class_pal, class_names = get_map_and_palette(top_50, 'Class', is_pathway=False)
path_map, path_pal, path_names = get_map_and_palette(top_50, 'Pathway', is_pathway=True)

top_50['Class_ID'] = top_50['Class'].map(class_map)
top_50['Path_ID'] = top_50['Pathway'].map(path_map)

# =============================================================================
# 5. FIGURE DESIGN & AXIS SPECIFICATION
# =============================================================================
num_genes = len(top_50)
fig_height = max(8, num_genes * 0.35) 

# Multi-panel layout declaration configuring narrow metadata panels flanking a wide horizontal bar plot
fig, (ax_class, ax_path, ax_bar) = plt.subplots(1, 3, figsize=(18, fig_height), 
                                                gridspec_kw={'width_ratios': [0.4, 0.4, 10]})

plt.subplots_adjust(wspace=0.05, bottom=0.06, top=0.94)

# -----------------------------------------------------------------------------
# Subplot 5a: Functional Class Sidebar Map
# -----------------------------------------------------------------------------
sns.heatmap(top_50[['Class_ID']].values, annot=False, cbar=False, 
            cmap=ListedColormap(class_pal), ax=ax_class)
ax_class.set_yticks([x + 0.5 for x in range(num_genes)])
ax_class.set_yticklabels(top_50['Gene'].values, rotation=0, fontsize=16) 
ax_class.set_title('Class', fontsize=18)
ax_class.set_xticks([])

# -----------------------------------------------------------------------------
# Subplot 5b: Metabolic Pathway Sidebar Map
# -----------------------------------------------------------------------------
sns.heatmap(top_50[['Path_ID']].values, annot=False, cbar=False, 
            cmap=ListedColormap(path_pal), ax=ax_path)
ax_path.set_yticks([]) 
ax_path.set_title('Path', fontsize=18) 
ax_path.set_xticks([])

# -----------------------------------------------------------------------------
# Subplot 5c: Horizontal Quantitative Expression Profile
# -----------------------------------------------------------------------------
# Dichotomous color array generation mapping up/down expression phenotypes
bar_colors = ['#d62728' if x < 0 else '#1f77b4' for x in top_50['Log Fold Change']]
y_positions = [x + 0.5 for x in range(num_genes)]
ax_bar.barh(y_positions, top_50['Log Fold Change'].values, color=bar_colors, height=0.7)

ax_bar.set_ylim(num_genes, 0) 
ax_bar.axvline(0, color='black', linewidth=1.5)
ax_bar.set_yticks([]) 
ax_bar.set_xlabel('Log2 Fold Change (Bone Marrow vs Peripheral Blood)', fontsize=18) 
ax_bar.tick_params(axis='x', labelsize=16)
ax_bar.set_title('Differential Glycogene Expression', fontsize=35, pad=15) 

# Configure grid matrix positioning to rest beneath geometric bar entities
ax_bar.set_axisbelow(True)
ax_bar.grid(axis='x', linestyle='--', alpha=0.3)

# =============================================================================
# 6. EXPANDEABLE LEGEND GENERATION
# =============================================================================
# Construct proxy patches to dynamically construct customized categorical legends
class_legends = [Patch(facecolor=class_pal[i], label=class_names[i]) for i in range(len(class_names))]
path_legends = [Patch(facecolor=path_pal[i], label=path_names[i]) for i in range(len(path_names))]

# Primary Legend anchoring: Glycogene Sub-classes
leg1 = ax_bar.legend(handles=class_legends, title="Class", title_fontsize=18,
                    loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False, fontsize=18)
ax_bar.add_artist(leg1)

# Secondary Legend anchoring: Downward layout offset tracking to prevent label overlap
legend_drop = 1 - (len(class_names) * 0.028) - 0.03
ax_bar.legend(handles=path_legends, title="Path", title_fontsize=18,
              loc='upper left', bbox_to_anchor=(1.02, legend_drop), frameon=False, fontsize=18)

# Output final compiled render pipeline
plt.show()
