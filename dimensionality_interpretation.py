import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import stats
from itertools import groupby
import functools
from typing import Optional
import logging
main_logger = logging.getLogger(__name__)

sns.set_theme(style="white")

def dim_rdxn_analysis(preds_df, categories, save_path=None):
    """
    Performs dimensionality reduction analysis (PCA + t-SNE) across multiple grouping categories.
    
    For each category (e.g., 'labels', 'gene', 'organism'), this function:
    - Generates a PCA heatmap with category group boundaries.
    - Computes one-way ANOVA tests to assess association between PCs and the category.
    - Plots a t-SNE 2D embedding colored by category.
    
    Additionally, it generates a t-SNE plot colored by 'probabilities', and merges significance
    results across all categories into a single DataFrame.

    Args:
        preds_df (pd.DataFrame): DataFrame containing columns 'inputs' (vectors) and categorical labels.
        categories (List[str]): List of column names to group by (e.g., ['labels', 'gene']).
        save_path (str, optional): If provided, saves plots and the merged PCA significance results.

    Returns:
        pd.DataFrame: Combined PCA-ANOVA results for all input categories, merged by PCA component.
    """
    pca_sig_dict = {}
    for category in categories:
        pca_results = plot_embeddings_pca_heatmap_with_groups(preds_df, n_components=50, category=category, save_path=save_path, logger=main_logger)
        pca_sig_dict[category] = test_association_of_pcs(pca_results, category=category)
        plot_embeddings_tsne(preds_df, category=category, save_path=save_path)
    plot_embeddings_tsne(preds_df, category='probabilities', save_path=save_path)
    pca_significance_df = functools.reduce(lambda left, right: pd.merge(left, right, on="Component", how="outer"),pca_sig_dict.values())
    if save_path:
        pca_significance_df.to_csv((f'{save_path}/pca_significance.csv'))
    return pca_significance_df
    
def convert_vector(x):
    """
    Converts a vector stored as a string or iterable into a NumPy array.
    
    Args:
        x: A vector represented as a string (e.g., "[0.1, 0.2, ...]") or an iterable.
        
    Returns:
        np.array: The vector as a NumPy array.
    """
    if isinstance(x, str):
        return np.array(ast.literal_eval(x))
    else:
        return np.array(x)


def get_tsne_df(vectors, category, df,  random_state, perplexity, learning_rate):
    """
    Runs t-SNE on a matrix of input vectors and returns a 2D embedding DataFrame.

    Args:
        vectors (np.ndarray): Matrix of shape (n_samples, n_features) containing input vectors.
        category (str): Column in `df` to retain for coloring (e.g., 'labels' or 'category').
        df (pd.DataFrame): Original DataFrame containing the category column.
        random_state (int): Seed for reproducibility.
        perplexity (float): t-SNE perplexity parameter.
        learning_rate (float): t-SNE learning rate.

    Returns:
        pd.DataFrame: DataFrame with columns ['TSNE1', 'TSNE2'] and the specified `category`.
    """
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=perplexity, learning_rate=learning_rate)
    embeddings_2d = tsne.fit_transform(vectors)
    tsne_df = pd.DataFrame(embeddings_2d, columns=['TSNE1', 'TSNE2'])
    tsne_df[category] = df[category].values
    return tsne_df

def plot_embeddings_tsne(df: pd.DataFrame, 
                         category= 'category', 
                         random_state: int = 42,
                         perplexity: float = 40.0, 
                         learning_rate: float = 200.0, 
                         palette='deep', 
                         hue_order=None,
                         legend_outside: bool = False, 
                         save_path = None) -> None:
    """
    Applies t-SNE dimensionality reduction on high-dimensional embedding vectors and plots a 2D scatter plot.
    Points are colored by the category specified in the 'category' column of the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with at least two columns:
            - 'inputs': High-dimensional embedding vector (list, np.array, or string representation).
            - 'category': Group indicator for each sample (e.g., 'TP', 'FP', etc.).
        random_state (int): Random seed for t-SNE (default: 42).
        perplexity (float): t-SNE perplexity parameter (default: 30.0).
        learning_rate (float): t-SNE learning rate (default: 200.0).
    
    Returns:
        None. Displays the t-SNE scatter plot.
    """
    vectors = np.vstack(df['inputs'].apply(convert_vector).values)
    tsne_df = get_tsne_df(vectors, category, df, random_state, perplexity, learning_rate)
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='TSNE1', y='TSNE2', hue=category, data=tsne_df, 
                    palette=palette, 
                    hue_order=hue_order, alpha=0.6)
    plt.title("t-SNE Visualization of Embedding Vectors",fontsize=16)
    plt.xlabel("TSNE Dimension 1",fontsize=16)
    plt.ylabel("TSNE Dimension 2",fontsize=16)
    plt.legend(title=category)
    if legend_outside:
        plt.legend(title="Category", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.legend(title="Category")
    sns.despine()
    plt.tight_layout()
    if save_path:
        plt.savefig(f'{save_path}/{category}_tsne.png', dpi=300)
        main_logger.info(f"Plot saved to {save_path}/{category}_tsne.png")
    else:
        plt.show()
    plt.close()


def get_pca(vectors_for_pca, n_components, random_state):
    """
    Applies PCA to the input vectors and returns the reduced components as a DataFrame.

    Args:
        vectors_for_pca (np.ndarray): Array of shape (n_samples, n_features) excluding the final logit.
        n_components (int): Number of principal components to retain.
        random_state (int): Seed for reproducibility.

    Returns:
        pd.DataFrame: DataFrame with PCA components named PC1, PC2, ..., PCn.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    embeddings_reduced = pca.fit_transform(vectors_for_pca)

    pca_columns = [f'PC{i+1}' for i in range(n_components)]
    pca_df = pd.DataFrame(embeddings_reduced, columns=pca_columns)
    return pca_df

def add_group_boundaries_and_labels_readable(
    ax: plt.Axes,
    ordered_groups: pd.Series,
    *,
    line_width: float = 3.0,
    label_offset: float = 0.02,
    fontsize: int = 12
) -> None:
    """
    Draw horizontal lines between contiguous groups and place a label at each group's vertical center.
    `ordered_groups` must be sorted in the same order as the heatmap rows.
    """
    nrows = len(ordered_groups)
    y = 0
    for name, grp in groupby(ordered_groups):
        size = len(list(grp))
        y_top = y
        y_bot = y + size
        y_mid = y_top + size / 2.0

        ax.hlines([y_top, y_bot], xmin=0, xmax=ax.get_xlim()[1], colors="white",
                  linewidth=line_width, zorder=3)

        ax.text(-label_offset * ax.get_xlim()[1], y_mid, str(name),
                va="center", ha="right", fontsize=fontsize, fontweight="bold")

        y += size

    ax.set_zorder(2)

def plot_embeddings_pca_heatmap_with_groups(
    df: pd.DataFrame, 
    random_state: int = 42, 
    n_components: int = 50,
    category: str = 'category',
    line_width: float = 4, 
    save_path: Optional[str] = None,
    *,
    title_prefix: str = "PCA Heatmap of Embedding Vectors with",
    ylabel_fontsize: int = 12,
    xtick_fontsize: int = 10,
    ylabel_size: int = 0,
    tick_pad_left: float = 0.30,
    logger: Optional[object] = None
) -> pd.DataFrame:
    """
    Reduce embeddings by PCA (keeping final element as 'LLR'), plot a readable heatmap,
    add clear group boundaries/labels, and log/print the ordered labels with counts.
    Returns the sorted DataFrame used to plot.
    """
    vectors = np.vstack(df['inputs'].apply(convert_vector).values)
    if vectors.shape[1] < 2:
        raise ValueError("Each vector must have at least two elements.")

    final_elements = vectors[:, -1].reshape(-1, 1)
    vectors_for_pca = vectors[:, :-1]
    n_comp = int(min(n_components, vectors_for_pca.shape[1]))

    pca_df = get_pca(vectors_for_pca, n_comp, random_state)  # expects PC1..PCn columns
    pca_df['LLR'] = final_elements.flatten()
    pca_df[category] = df[category].values

    pca_df_sorted = pca_df.sort_values(by=category, kind="stable").reset_index(drop=True)
    heatmap_data = pca_df_sorted.drop(columns=[category])

    fig, ax = plt.subplots(figsize=(18, 10))
    sns.heatmap(
        heatmap_data,
        cmap='viridis',
        ax=ax,
        cbar_kws={'label': ''},
        yticklabels=False,
        xticklabels=True
    )

    plt.subplots_adjust(left=tick_pad_left, bottom=0.18, right=0.95, top=0.90)

    ax.set_xlabel("Features (PCA components + final element)")
    ax.set_ylabel("")
    ax.tick_params(axis='x', labelsize=xtick_fontsize)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

    plt.title(f"{title_prefix} {category} Labels", fontsize=16, pad=12)

    add_group_boundaries_and_labels_readable(
        ax, pca_df_sorted[category],
        line_width=line_width,
        label_offset=0.04,
        fontsize=12
    )

    ax.set_axisbelow(True)
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)

    if save_path:
        out = f'{save_path}/{category}_pca.png'
        plt.savefig(out, dpi=300)
        if logger: logger.info(f"Plot saved to {out}")
    else:
        plt.show()
    plt.close(fig)

    # ---- log/print ordered labels with counts ----
    ordered_labels = []
    for name, grp in groupby(pca_df_sorted[category]):
        count = len(list(grp))
        ordered_labels.append((str(name), count))

    msg_lines = ["Group order (top→bottom):"] + [f"- {n} (n={c})" for n, c in ordered_labels]
    msg = "\n".join(msg_lines)
    if logger:
        logger.info(msg)
    else:
        print(msg)

    return pca_df_sorted


def test_association_of_pcs(pca_df: pd.DataFrame, category: str = 'category') -> pd.DataFrame:
    """
    Tests whether each principal component (and the 'LLR' column, if present) is significantly associated 
    with the groups defined in the given category. For each column, a one-way ANOVA is performed comparing the 
    means across groups.

    Args:
        pca_df (pd.DataFrame): DataFrame containing PCA components and the group indicator.
        category (str): Column name in pca_df that indicates group membership.

    Returns:
        pd.DataFrame: A DataFrame with columns ['Component', f'F_value_{category}', f'p_value_{category}', f'significant_{category}'].
    """

    components = [col for col in pca_df.columns if col.startswith("PC")]
    if "LLR" in pca_df.columns:
        components.append("LLR")
    
    groups = pca_df[category].unique()
    if len(groups) < 2:
        main_logger.info(f"Skipping ANOVA: only one unique group in '{category}'")
        # Return empty DataFrame with proper columns
        return pd.DataFrame(columns=["Component", f'F_value_{category}', f'p_value_{category}', f'significant_{category}'])
    
    results = []
    
    for comp in components:
        group_data = [pca_df.loc[pca_df[category] == g, comp].values for g in groups]
        if len(group_data) < 2 or any(len(data) < 2 for data in group_data):
            main_logger.info(f"Skipping component '{comp}': insufficient samples in one or more groups.")
            continue
        
        try:
            f_val, p_val = stats.f_oneway(*group_data)
        except Exception as e:
            main_logger.info(f"Skipping component '{comp}' due to error in f_oneway: {e}")
            continue
        
        results.append({
            'Component': comp, 
            f'F_value_{category}': f_val, 
            f'p_value_{category}': p_val
        })
    
    results_df = pd.DataFrame(results, columns=["Component", f'F_value_{category}', f'p_value_{category}'])
    if not results_df.empty:
        results_df = results_df.sort_values(by=f'p_value_{category}', ascending=True).reset_index(drop=True)
        results_df[f'significant_{category}'] = results_df[f'p_value_{category}'] < 0.01
    else:
        results_df = pd.DataFrame(columns=["Component", f'F_value_{category}', f'p_value_{category}', f'significant_{category}'])
        main_logger.info("No components were analyzed successfully.")
    
    return results_df

