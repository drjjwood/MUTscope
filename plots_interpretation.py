import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import py3Dmol
import time
import logging
from matplotlib import cm, colors
main_logger = logging.getLogger(__name__)

sns.set_theme(style="white")

def plot_hist_probabilities(df, save_path=None):
    """
    Plots a density histogram of predicted probabilities (single distribution).

    Args:
        df (pd.DataFrame): Input dataframe with 'probabilities' column.
        save_path (str, optional): If provided, saves the plot to this path.

    Returns:
        None
    """
    sns.histplot(data=df, x='probabilities', hue = "binary_predictions",
                 stat='density', palette=['#cc79a7','#0072b2'])
    sns.despine()
    if save_path:
        plt.savefig(f'{save_path}/probability_histogram.png')
        main_logger.info(f"Plot saved to {save_path}/probability_histogram.png")
    else:
        plt.show()
    plt.close()

def category_histplot(df, x_col, hue_col, hue_order=None, save_path=None):
    """
    Plots a normalized stacked histogram (proportion per bar) of a categorical x-axis 
    grouped by a hue column (e.g., TP, TN, FP, FN).

    Args:
        df (pd.DataFrame): Input dataframe.
        x_col (str): Column name for x-axis (e.g., gene or organism).
        hue_col (str): Column name for classification group (e.g., 'category').
        hue_order (list, optional): Manual order for hue categories.
        save_path (str, optional): If provided, saves the plot to this path.

    Returns:
        None
    """
    plt.figure()
    ax = sns.histplot(
        data=df,x=x_col, hue=hue_col,
        multiple="fill", stat="proportion",
        discrete=True, shrink=.8, 
        palette=['#cc79a7','#0072b2','#f0e442','#009e73'],
        hue_order=hue_order
        )
    sns.move_legend(ax,"upper right", bbox_to_anchor=(1.35, 1))
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f'{save_path}/{x_col}_category_histplot.png')
        main_logger.info(f"Plot saved to {save_path}/{x_col}_category_histplot.png")
    else:
        plt.show()
    plt.close()
    
def plot_histogram_with_categories(df, x_col, threshold=0.3, common_norm=True, save_path=None):
    """
    Plots a probability histogram split by classification category (TP, TN, FP, FN),
    with an optional decision threshold overlay.

    Args:
        df (pd.DataFrame): Input dataframe with 'category' column.
        x_col (str): Column name for the x-axis (e.g., 'probabilities').
        threshold (float, optional): Threshold to draw a vertical line. Default is 0.3.
        common_norm (bool, optional): Whether to normalize histograms to the same total.
        save_path (str, optional): If provided, saves the plot to this path.

    Returns:
        None
    """
    plt.figure()
    sns.histplot(data=df, x=x_col, hue = "category",stat='density', 
                 palette=['#cc79a7','#0072b2','#f0e442','#009e73'], hue_order=['TN','TP','FN','FP'],common_norm=common_norm)
    plt.axvline(x=threshold, color='grey', linestyle='--')
    sns.despine()

    if save_path:
        plt.savefig(f'{save_path}/{x_col}_category_histogram.png')
        main_logger.info(f"Plot saved to {save_path}/{x_col}_category_histogram.png")
    else:
        plt.show()
    plt.close()


# ----below are acessory notebook fxns -----

def plot_scatterplot(data_df, x_col, y_col, hue_col=None, col_pallette="tab20", ax=None, correlation=False, figure_size=(12, 4)):
    """
    Plots a scatterplot of two columns in the provided DataFrame.
    
    Optionally, colors points by a hue column and annotates the plot with the Spearman correlation coefficient.

    Args:
        data_df (pd.DataFrame): DataFrame containing the data.
        x_col (str): Column name for the x-axis.
        y_col (str): Column name for the y-axis.
        hue_col (str, optional): Column name for grouping/coloring data points. Defaults to None.
        col_pallette (str, optional): Seaborn palette name. Defaults to "tab20".
        ax (matplotlib.axes.Axes, optional): Axes object on which to plot. If None, a new figure and axes are created.
        correlation (bool, optional): If True, computes and annotates the plot with Spearman's r. Defaults to False.
        figure_size (tuple, optional): Figure size if ax is None. Defaults to (12, 4).

    Returns:
        matplotlib.axes.Axes: The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figure_size)
    sns.scatterplot(data=data_df, x=x_col, y=y_col, hue=hue_col, alpha=0.5, palette=col_pallette, ax=ax)
    ax.set_title(f'{y_col} vs {x_col}')
    if correlation:
        spr = stats.spearmanr(data_df[x_col], data_df[y_col], nan_policy='omit')[0]
        ax.annotate(f'Spearman r={spr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, verticalalignment='top')
    sns.despine(ax=ax)
    return ax


def plot_boxplot(data_df, x_col, y_col, hue_col=None, col_pallette="tab20", ax=None, t_test=True):
    """
    Plots a boxplot of a continuous variable against a categorical variable from the DataFrame.
    
    Optionally, performs a t-test between the groups and annotates the plot with the test statistic and p-value.

    Args:
        data_df (pd.DataFrame): DataFrame containing the data.
        x_col (str): Column name for the categorical variable (x-axis).
        y_col (str): Column name for the continuous variable (y-axis).
        hue_col (str, optional): Additional grouping variable for box colors. Defaults to None.
        col_pallette (str, optional): Seaborn palette name. Defaults to "tab20".
        ax (matplotlib.axes.Axes, optional): Axes object on which to plot. If None, a new figure and axes are created.
        t_test (bool, optional): If True, performs an independent two-sample t-test and annotates the plot. Defaults to True.

    Returns:
        matplotlib.axes.Axes: The axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    sns.boxplot(data=data_df, x=x_col, y=y_col, hue=hue_col, palette=col_pallette, ax=ax)
    ax.set_title(f'{y_col} vs {x_col}')
    if t_test:
        ttest = get_ttest_stats(data_df, x_col, y_col)
        ax.annotate(f't-test stat={ttest[0]:.2f}, p-val={ttest[1]:.2f}', xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10, verticalalignment='top')
        main_logger.info(f'{y_col} by {x_col} t-test stat={ttest[0]:.2f}, p-val={ttest[1]:.2f}')
    sns.despine(ax=ax)
    return ax


def get_ttest_stats(data_df, binary_col, continuous_col):
    """
    Performs an independent two-sample t-test comparing the values of a continuous variable 
    between two groups defined by a binary categorical column.

    Args:
        data_df (pd.DataFrame): DataFrame containing the data.
        binary_col (str): Column name for the binary grouping variable.
        continuous_col (str): Column name for the continuous variable.

    Returns:
        tuple: t-statistic and p-value from the t-test.
    
    Raises:
        ValueError: If the binary column does not contain exactly two unique values.
    """
    options = list(data_df[binary_col].unique())
    if len(options) != 2:
        raise ValueError(f"Binary column '{binary_col}' must have exactly two unique values, found {len(options)}.")
    dataset_1 = data_df[data_df[binary_col] == options[0]]
    dataset_2 = data_df[data_df[binary_col] == options[1]]
    return stats.ttest_ind(dataset_1[continuous_col], dataset_2[continuous_col], nan_policy='omit')


def create_plot_grid(data_df, x_col, features, hue_col=None, col_pallette="tab20", ncols=3, plot_type='scatter'):
    """
    Creates a grid of plots for multiple features against a specified x-axis column.

    Args:
        data_df (pd.DataFrame): DataFrame containing the data.
        x_col (str): Column name to use for the x-axis in all plots.
        features (list of str): List of column names for the y-axis of each subplot.
        hue_col (str, optional): Column name for additional grouping. Defaults to None.
        col_pallette (str, optional): Seaborn palette name. Defaults to "tab20".
        ncols (int, optional): Number of columns in the grid. Defaults to 3.
        plot_type (str, optional): Type of plot to generate: 'scatter' or 'box'. Defaults to 'scatter'.

    Returns:
        None. Displays the grid of plots.
    """
    nrows = -(-len(features) // ncols) 
    fig, axs = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows))
    axs = axs.flatten()  # Flatten the array of axes for easy iteration

    for ax, feature in zip(axs, features):
        if plot_type == 'scatter':
            plot_scatterplot(data_df, x_col, feature, hue_col, col_pallette, ax=ax)
        elif plot_type == 'box':
            plot_boxplot(data_df, x_col, feature, hue_col, col_pallette, ax=ax)
        else:
            raise ValueError("plot_type must be either 'scatter' or 'box'.")
    
    # Hide any remaining empty subplots.
    for ax in axs[len(features):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_probs_along_seq(df,
                         uniprot_id,
                         feature=None,
                         threshold=0.3,
                         fill_missing=False,
                         mode='both'):
    """
    Plot model‐predicted probabilities (and an optional binary feature)
    along the amino‐acid positions of a UniProt sequence, with three modes:
    
      mode='lines'   → only the mean lines
      mode='points'  → only the raw points
      mode='both'    → both lines and points
    
    Arguments:
      • df: DataFrame with columns ['uniprot_id','position','wt_sequence',
             'probabilities', feature]
      • uniprot_id: string, which protein to plot
      • feature:        string or None, name of a 0/1 column to overlay
      • threshold:      float, y-value for a horizontal dashed line
      • fill_missing:   bool, if True the line-plots see zeros at missing positions
      • mode:           'lines' | 'points' | 'both'
    """
    # 1) real observations
    df_raw = (
        df[df['uniprot_id'] == uniprot_id]
        .sort_values('position')
        .reset_index(drop=True)
    )
    if df_raw.empty:
        raise ValueError(f"No data for {uniprot_id}")

    # 2) get sequence length
    seqs = df_raw['wt_sequence'].drop_duplicates()
    if len(seqs) != 1:
        print(f"Warning: {len(seqs)} wt_sequences found; using the first.")
    seq = seqs.iloc[0]
    ax_len = len(seq)

    # 3) line-data with optional fill
    df_line = df_raw.copy()
    if fill_missing:
        full = pd.DataFrame({'position': np.arange(1, ax_len+1)})
        df_line = full.merge(df_line, on='position', how='left')
        df_line['probabilities'] = df_line['probabilities'].fillna(0)
        if feature:
            df_line[feature] = df_line[feature].fillna(0)

    sns.set(style="whitegrid")
    plt.figure(figsize=(12,6))

    # --- PROBABILITIES ---
    if mode in ('points','both'):
        jitter_p = (np.random.rand(len(df_raw)) - 0.5) * 0.4
        plt.scatter(
            df_raw['position'] + jitter_p,
            df_raw['probabilities'],
            color='#0072b2',
            alpha=0.4,
            s=20,
            edgecolor='none',
            label='Raw probabilities'
        )
    if mode in ('lines','both'):
        sns.lineplot(
            data=df_line,
            x='position', y='probabilities',
            estimator='mean',
            errorbar=None,
            color='#0072b2',
            label='Probability (mean)'
        )

    # --- OPTIONAL BINARY FEATURE ---
    if feature:
        if mode in ('points','both'):
            jitter_f = (np.random.rand(len(df_raw)) - 0.5) * 0.4
            plt.scatter(
                df_raw['position'] + jitter_f,
                df_raw[feature],
                color='#cc79a7',
                alpha=0.4,
                s=20,
                edgecolor='none',
                label=f'Raw {feature}'
            )
        if mode in ('lines','both'):
            sns.lineplot(
                data=df_line,
                x='position', y=feature,
                estimator='mean',
                errorbar=None,
                color='#cc79a7',
                label=f'{feature} (mean)'
            )

    # --- THRESHOLD LINE ---
    plt.axhline(
        y=threshold,
        color='#f0e442',
        linestyle='--',
        label='Threshold'
    )

    # finalize
    plt.xlim(1, ax_len)
    plt.xlabel("Position", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    title_feat = f" + {feature}" if feature else ""
    plt.title(f"Probabilities{title_feat} along sequence for {uniprot_id}", fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_structure_continuous_color(pdb_file: str, 
                                              value_df: pd.DataFrame, 
                                              value_col: str,
                                              position_col: str = 'position',
                                              chain: str = None, 
                                              default_color: str = "#add8e6", 
                                              colormap: str = "cool", 
                                              output_file: str = None) -> object:
    """
    Visualize a PDB structure as a cartoon (ribbon) and color residues based on a continuous variable.
    
    Args:
        pdb_file (str): Path to the PDB file.
        value_df (pd.DataFrame): DataFrame containing residue-level values. Must contain:
            - position_col: Residue number (numeric, e.g., integer). 
            - value_col: Continuous variable (float) to map to color.
        position_col (str): Name of the column that contains residue positions.
        value_col (str): Name of the column that contains the continuous variable to map.
        chain (str, optional): If provided, only residues in this chain will be colored according to value.
        default_color (str, optional): Default color (hex) for residues without a provided value.
        colormap (str, optional): Name of the matplotlib colormap to use.
        output_file (str, optional): If provided, saves a PNG image of the view to this file.
    
    Returns:
        view: A py3Dmol view object for interactive visualization.
    """
    with open(pdb_file, 'r') as file:
        pdb_data = file.read()
    
    view = py3Dmol.view(width=800, height=600)
    view.addModel(pdb_data, 'pdb')
    
    view.setStyle({'cartoon': {}}, {'cartoon': {'color': default_color}})
    
    # Determine the normalization from the value column.
    if not value_df.empty and value_col in value_df.columns:
        vmin = value_df[value_col].min()
        vmax = value_df[value_col].max()
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = cm.get_cmap(colormap)
        
        # Create a mapping from residue number to hex color.
        color_mapping = {}
        for _, row in value_df.iterrows():
            pos_val = row[position_col]
            # Skip rows where position is NaN.
            if pd.isna(pos_val):
                continue
            try:
                pos = int(pos_val)
            except Exception as e:
                print(f"Skipping row due to position conversion error: {e}")
                continue
            val = row[value_col]
            rgba = cmap(norm(val))
            hex_color = colors.rgb2hex(rgba)
            color_mapping[pos] = hex_color
    else:
        color_mapping = {}
    
    # Loop over each residue position in the mapping and set its style.
    for pos, hex_color in color_mapping.items():
        # Build the selection. If a chain is specified, restrict the selection to that chain.
        selection = {'resi': pos}
        if chain is not None:
            selection['chain'] = chain
        # Overwrite the cartoon color for this residue.
        view.setStyle(selection, {'cartoon': {'color': hex_color}})
 
    view.setBackgroundColor('black')
    view.zoomTo()
    
    view.show()
    
    if output_file:
        png_data = view.png()
        with open(output_file, 'wb') as f:
            f.write(png_data)
        print(f"Image saved as {output_file}")
    
    # Allow time for the view to render.
    time.sleep(5)
    return view
