import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from matplotlib.patches import Patch
import matplotlib.patheffects as pe
from Bio import PDB
from matplotlib import colors

def plot_hist_with_threshold(data, threshold=0.3, bins=100,
                             title=f"DMS Probabilities\n",
                             xlabel="Probability",
                             ylabel="Density",
                             figsize=(10, 6),
                             text_size=30,
                             highlight_color="#FDE725FF",
                             default_color="#414487FF",  # default bar color
                             alpha=0.8,
                             line_style="--",
                             line_color="gray"):
    """
    Plot a histogram of `data` with density normalization, highlight bars above a given threshold,
    add a vertical dashed threshold line, and label everything with large font sizes.
    Bars are drawn with specified alpha, and any existing gridlines are removed.
    """
    counts, bin_edges = np.histogram(data, bins=bins, density=True)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges[:-1] + bin_width / 2

    bar_colors = [highlight_color if center > threshold else default_color
                  for center in bin_centers]

    fig, ax = plt.subplots(figsize=figsize)
    ax.grid(False)

    ax.bar(bin_centers, counts, width=bin_width,
           color=bar_colors, edgecolor='black', alpha=alpha)

    line = ax.axvline(threshold, linestyle=line_style, color=line_color, linewidth=2)
    
    handles, labels = ax.get_legend_handles_labels()
    
    resistant_patch = Patch(facecolor='#FDE725FF', edgecolor='black', label='Resistant')
    sensitive_patch = Patch(facecolor='#414487FF', edgecolor='black', label='Sensitive')
    
    handles.extend([resistant_patch, sensitive_patch])
    labels.extend(['Resistant', 'Sensitive'])
    
    ax.legend(
        handles=handles,
        title="Prediction",
        labels=labels,
        bbox_to_anchor=(1.02,1),
        loc="upper left",
        fontsize=20,
        title_fontsize=20
    )

    ax.set_title(title, fontsize=text_size)
    ax.set_xlabel(xlabel, fontsize=text_size)
    ax.set_ylabel(ylabel, fontsize=text_size)

    ax.tick_params(axis='both', labelsize=text_size)
    
    plt.tight_layout()
    sns.despine()
    fig.savefig(f'probabilities_{selected_org}_{selected_gene}.png', dpi=300, bbox_inches='tight')
    plt.show()

def assign_risk_labels(y_proba, bins=[0.0, 0.1, 0.3, 0.6, 0.9, 1.0],
                       labels=["Very Low", "Low", "Medium", "High", "Very High"]):
    """
    Assigns risk labels to each probability based on predefined bins.
    
    Args:
        y_proba (array-like): Predicted probabilities
        bins (list): Bin edges for probability intervals
        labels (list): Corresponding labels for bins

    Returns:
        List of risk labels
    """
    bin_indices = np.digitize(y_proba, bins, right=False) - 1
    bin_indices = np.clip(bin_indices, 0, len(labels) - 1)  # Just in case
    return [labels[i] for i in bin_indices]

def plot_risk_hist(
    probs: np.ndarray,
    selected_gene,
    selected_org,
    hist_bins: int = 100,
    coarse_bins: list = [0.0, 0.1, 0.3, 0.6, 0.9, 1.0],
    risk_labels: list = ["very low", "low", "medium", "high", "very high"],
    threshold: float = 0.3,
    density: bool = False,
    figsize=(8,5),
    cmap: str = "viridis",
    alpha: float = 0.8,
    legend_title: str = "Risk",
):
    """
    Plot a fine‐binned histogram of `probs`, color each bar by which
    coarse‐bin its center falls into, and show a ‘Risk’ legend with
    custom labels and a dashed threshold line.
    """
    probs = np.asarray(probs)
    n_coarse = len(coarse_bins) - 1
    assert len(risk_labels) == n_coarse, "risk_labels must have len(coarse_bins)-1"

    counts, edges = np.histogram(probs, bins=hist_bins, density=density)
    widths = np.diff(edges)
    centers = edges[:-1] + widths/2

    palette = sns.color_palette(cmap, n_coarse)

    idx = np.digitize(centers, bins=coarse_bins, right=False) - 1
    idx = np.clip(idx, 0, n_coarse-1)

    fig, ax = plt.subplots(figsize=figsize)
    for i in range(len(centers)):
        ax.bar(
            centers[i], counts[i],
            width=widths[i],
            color=palette[idx[i]],
            alpha=alpha,
            edgecolor="none"
        )

    ax.axvline(threshold, color="gray", linestyle="--", linewidth=2)

    handles = []
    for j in range(n_coarse):
        handles.append(
            Patch(facecolor=palette[j], edgecolor="none", label=risk_labels[j])
        )
    handles.append(
        plt.Line2D([0],[0], color="gray", linestyle="--", lw=2, label=f"threshold = {threshold}")
    )
    ax.legend(
        handles=handles,
        fontsize=16, 
        loc="lower center",
        bbox_to_anchor=(0.5, -0.25),
        ncol=len(handles),
        frameon=False
    )

    ax.set_xlabel("Probability", fontsize=18)
    ax.set_ylabel("Density" if density else "Count", fontsize=18)
    ax.tick_params(axis='both', labelsize=16) 

    sns.despine()
    plt.tight_layout()
    fig.savefig(f'risk_hist_{selected_org}_{selected_gene}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return fig, ax

def plot_positions_hm(position_df, risk_group, selected_org, selected_gene):
    """
    Heatmap of per‐position probabilities, with each cell annotated
    by its numeric value and a colour bar.
    """
    data = position_df.T
    annot_kws = {
    "fontsize": 14,
    "color":   "white",
    "path_effects": [ pe.withStroke(linewidth=2, foreground="black") ]
    }

    fig, ax = plt.subplots(figsize=(20, 2))
    sns.heatmap(
        data,
        cmap="viridis",
        linewidths=0.1,
        linecolor='grey',
        vmin=data.min().min(),
        vmax=data.max().max(),
        cbar=True,
        annot=True,               # turn on annotations
        fmt=".2f",                # format each number to 2 decimal places
        annot_kws=annot_kws,
        ax=ax
    )

    ax.set_ylabel("Position", fontsize=14)
    ax.set_xlabel("Amino Acid", fontsize=14)
    ax.set_title(f"{risk_group} Position Probability Heatmap", fontsize=14)

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout()
    fig.savefig(f'position_hm_{risk_group}_{selected_org}_{selected_gene}.png', dpi=300, bbox_inches='tight')
    plt.show()

def prepare_full_df(preds_path, original_data_path, wt, selected_org, selected_gene, protein_id):
    preds = pd.read_csv(preds_path)
    preds = prepare_preds_df(preds, protein_id)
    original_data = pd.read_csv(original_data_path)
    subset_data = prepare_subset_df(original_data, selected_org, selected_gene)                                                                      
    full_data = preds.merge(subset_data, right_on = 'protein_change',left_on = 'protein_change', how = 'left')
    full_data['wt_sequence'] = wt
    full_data['risk'] = assign_risk_labels(full_data['probabilities'])
    profile_data(full_data, selected_org, selected_gene)
    return full_data

def prepare_preds_df(preds, protein_id):
    preds['uniprot_id'] =protein_id
    preds = preds.merge(preds['target_id'].str.split('.', expand=True), right_index=True,  left_index=True)
    preds = preds.rename(columns={0:'organism',1:'gene',2:'protein_change'})
    preds['wt_aa'] = preds['protein_change'].str[0]
    preds['mt_aa'] = preds['protein_change'].str[-1]
    preds['position'] = preds['protein_change'].str[1:-1].astype(int)
    return preds

def prepare_subset_df(original_data, selected_org, selected_gene):
    original_data['tag'] = 'original_dataset'
    subset_data = original_data[(original_data['gene']==selected_gene)&(original_data['organism']==selected_org)][['protein_change','alphafold_position','label','tag','antibiotic_class',  'antibiotic_subclass','antibiotic_codes','antibiotic_standardised']] 
    return subset_data

def profile_data(full_data, selected_org, selected_gene):
    print(f'For {selected_org} {selected_gene}:\n')
    antibiotics = full_data['antibiotic_subclass'].unique()
    print(f'Resistance mutations to: {antibiotics}\n')
    pos_muts = list(full_data[full_data['label']==1].sort_values(by='position')['protein_change'])
    print(f'N resistant mutations in original data: {len(pos_muts)}')
    print(f'Resistant mutations in original data: {pos_muts}\n')
    neg_muts = list(full_data[full_data['label']==0].sort_values(by='position')['protein_change'])
    print(f'N sensitive mutations in original data: {len(neg_muts)}')
    print(f'Sensitive mutations in original data: {neg_muts}\n')
    pred_counts = full_data['binary_predictions'].value_counts()
    print(f'Prediction counts: {pred_counts}')

def setup_heatmap_df(full_data, values):
    full_data['position'] = pd.to_numeric(full_data['position'], errors='coerce')
    heatmap_df = full_data.pivot(index='mt_aa', columns='position', values=values).fillna(0)
    heatmap_df.columns = heatmap_df.columns.astype(int)
    heatmap_df = heatmap_df.sort_index(axis=1)
    aa_order = list("ACDEFGHIKLMNPQRSTVWY")
    heatmap_df = heatmap_df.loc[heatmap_df.index.intersection(aa_order)]
    heatmap_df = heatmap_df.reindex(aa_order)
    return heatmap_df


def plot_dms_heatmap(full_data, values, selected_org, selected_gene, figsize=(30, 10)): 
    heatmap_df = setup_heatmap_df(full_data, values)
    fig, ax = plt.subplots(figsize=figsize)
    xticks = range(0, heatmap_df.shape[1], 20) # Show tick every 20th position
    xtick_labels = [str(heatmap_df.columns[i]) for i in xticks]
    
    sns.heatmap(
        heatmap_df,
        cmap="viridis",
        linewidths=0.1,
        linecolor='grey',
        vmin=0,
        vmax=1,
        cbar=True,
        ax=ax)
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, rotation=90, fontsize=36)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=36)
    ax.set_xlabel(f"\n Sequence Position", fontsize=36)
    ax.set_ylabel(f"Amino Acid\n ", fontsize=36)
    
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=30)
    
    plt.tight_layout()
    plt.show()
    fig.savefig(f'heatmap_{values}_{selected_org}_{selected_gene}.png', dpi=300, bbox_inches='tight')
    return heatmap_df

def export_colored_pdb(pdb_file, value_df, value_col, position_col="position",
                       chain="A", colormap="viridis", out_pdb="colored.pdb"):
    """
    Reads pdb_file, writes out a new PDB with residue values encoded into the B-factor column.
    This can be opened in PyMOL and colored by b-factor.
    """
    # load pdb
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure("prot", pdb_file)

    # normalize values to [0,1]
    vmin = float(value_df[value_col].min())
    vmax = float(value_df[value_col].max())
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    # mapping from residue position -> value
    valmap = dict(zip(value_df[position_col].astype(int), value_df[value_col]))

    for model in structure:
        for ch in model:
            if chain and ch.id != chain:
                continue
            for res in ch:
                res_id = res.get_id()[1]  # residue number
                if res_id in valmap:
                    val = float(valmap[res_id])
                    normed = norm(val)
                else:
                    normed = 0.0
                # set all atom B-factors = normalized value
                for atom in res:
                    atom.set_bfactor(normed * 100.0)  # scale to [0,100]

    io = PDB.PDBIO()
    io.set_structure(structure)
    io.save(out_pdb)
    print(f"Saved {out_pdb} with values in B-factor column")

















