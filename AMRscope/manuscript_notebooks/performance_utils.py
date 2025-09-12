import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch
from dataclasses import dataclass
from typing import Sequence, Mapping, Optional, Tuple, List

def reshape_summary(df, 
                    metrics,
                    avg_prefix="Average ",
                    std_prefix="Stdev "):
    """
    Transform a summary DataFrame with 'Average <model>' and 'Stdev <model>'
    rows into a long-form DataFrame with columns: model, metric_name, mean, sd.
    
    Parameters:
    - df: pd.DataFrame, must include a 'Name' column and metric columns.
    - metrics: list of metric column names; if None, infer all except 'Name'.
    - avg_prefix: prefix identifying average rows.
    - std_prefix: prefix identifying stdev rows.
    
    Returns:
    - pd.DataFrame with columns ['model','metric_name','mean','sd'].
    """
    avg = df[df['Name'].str.startswith(avg_prefix)].copy()
    std = df[df['Name'].str.startswith(std_prefix)].copy()
    
    avg['model'] = avg['Name'].str.replace(avg_prefix, '', regex=False)
    std['model'] = std['Name'].str.replace(std_prefix, '', regex=False)
    
    avg_long = avg.melt(id_vars=['model'],
                        value_vars=metrics,
                        var_name='metric_name',
                        value_name='mean')
    std_long = std.melt(id_vars=['model'],
                        value_vars=metrics,
                        var_name='metric_name',
                        value_name='sd')
    
    plot_df = pd.merge(avg_long, std_long, on=['model','metric_name'])
    return plot_df


@dataclass
class StyleConfig:
    figsize: Tuple[int, int] = (20, 8)
    context: str = "talk"
    font_size: int = 30
    xlabel: str = ""
    ylabel: str = "Mean ± SD"
    title: str = ""
    legend_title: str = "Model Type"
    rotate_xticks: int = 45
    grid_color: str = "#EEEEEE"
    grid_lw: float = 1.0
    ylim: Tuple[float, float] = (0.0, 1.0)
    yticks: Optional[List[float]] = None  # e.g., [0, 0.2, 0.4, 0.6, 0.8, 1]


@dataclass
class ErrorBarConfig:
    capsize: float = 5.0
    ecolor: str = "black"
    lw: float = 1.5
    sd_col: str = "sd"
    mean_col: str = "mean"


def compute_hue_order_by_metric(
    df: pd.DataFrame,
    metric_col: str = "metric_name",
    target_metric: str = "test_best_model_f1",
    mean_col: str = "mean",
) -> List[str]:
    """
    Return models ordered by descending mean for a specific metric.
    """
    f1_df = df[df[metric_col] == target_metric]
    if f1_df.empty:
        # Fall back to alphabetical if target metric not present
        return sorted(df["model"].unique().tolist())
    return (
        f1_df.sort_values(mean_col, ascending=False)["model"]
        .dropna()
        .tolist()
    )


def set_seaborn_context(style: StyleConfig) -> None:
    """
    Apply a consistent seaborn/matplotlib context and font sizes.
    """
    sns.set_context(
        style.context,
        rc={
            "axes.titlesize": style.font_size,
            "axes.labelsize": style.font_size,
            "xtick.labelsize": style.font_size,
            "ytick.labelsize": style.font_size,
            "legend.title_fontsize": style.font_size,
            "legend.fontsize": style.font_size,
        },
    )


def validate_inputs(
    df: pd.DataFrame,
    metrics_order: List[str],
    metric_col: str = "metric_name",
    mean_col: str = "mean",
    sd_col: str = "sd",
) -> None:
    """
    Sanity checks so the plot fails fast with helpful errors.
    """
    required_cols = {"model", metric_col, mean_col, sd_col}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"df is missing columns: {sorted(missing)}")

    missing_metrics = [m for m in metrics_order if m not in df[metric_col].unique()]
    if missing_metrics:
        print(f"[plot_summary] Warning: metrics not found in df: {missing_metrics}")


def draw_grouped_barplot(
    df: pd.DataFrame,
    metrics_order: List[str],
    metric_labels: List[str],
    palette,
    hue_order: List[str],
    style: StyleConfig,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Draw grouped bars (without error bars yet).
    """
    fig, ax = plt.subplots(figsize=style.figsize)
    sns.barplot(
        data=df,
        x="metric_name",
        y="mean",
        hue="model",
        order=metrics_order,
        hue_order=hue_order,
        palette=palette,
        dodge=True,
        errorbar=None,
        ax=ax,
    )

    x_vals = np.arange(len(metrics_order))
    ax.set_xticks(x_vals)
    ax.set_xticklabels(metric_labels, rotation=style.rotate_xticks, ha="right")
    ax.set_xlabel(style.xlabel)

    ax.set_ylim(*style.ylim)
    if style.yticks is not None:
        ax.set_yticks(style.yticks)
    ax.set_ylabel(style.ylabel)

    ax.set_title(style.title)
    ax.legend(
        title=style.legend_title,
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=style.grid_color, linewidth=style.grid_lw)

    return fig, ax


def add_grouped_errorbars(
    ax: plt.Axes,
    df: pd.DataFrame,
    metrics_order: List[str],
    hue_order: List[str],
    err_cfg: ErrorBarConfig,
) -> None:
    """
    Add manual error bars aligned to each group of bars.
    """
    if not ax.patches:
        return

    bar_width = ax.patches[0].get_width()
    n_models = len(hue_order)
    x_vals = np.arange(len(metrics_order))

    for i, model in enumerate(hue_order):
        subset = (
            df[df["model"] == model]
            .set_index("metric_name")
            .reindex(metrics_order)
        )
        offsets = (i - (n_models - 1) / 2) * bar_width
        ax.errorbar(
            x_vals + offsets,
            subset[err_cfg.mean_col].values,
            yerr=subset[err_cfg.sd_col].values,
            fmt="none",
            ecolor=err_cfg.ecolor,
            capsize=err_cfg.capsize,
            lw=err_cfg.lw,
        )


def plot_summary(
    df_plot: pd.DataFrame,
    metrics_order: List[str],
    metric_labels: List[str],
    palette,
    style: Optional[StyleConfig] = None,
    err_cfg: Optional[ErrorBarConfig] = None,
    order_by_metric: str = "test_best_model_f1",
    save_path: Optional[str] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a grouped bar chart with manual SD error bars.

    Parameters
    ----------
    df_plot : pd.DataFrame
        Long-form dataframe with columns: ['model', 'metric_name', 'mean', 'sd'].
    metrics_order : list[str]
        Order of metric_name categories along the x-axis.
    metric_labels : list[str]
        Pretty labels (same length as metrics_order).
    palette : list/str/dict
        Seaborn-compatible palette.
    style : StyleConfig, optional
        Figure/axes styling config.
    err_cfg : ErrorBarConfig, optional
        Error bar styling config.
    order_by_metric : str
        Metric to sort models by (descending mean).
    save_path : str, optional
        If provided, save the figure to this path (e.g., "summary.png").

    Returns
    -------
    (fig, ax) : Tuple[Figure, Axes]
        Matplotlib figure and axes for further customization.
    """
    style = style or StyleConfig()
    err_cfg = err_cfg or ErrorBarConfig()

    validate_inputs(df_plot, metrics_order, metric_col="metric_name",
                    mean_col=err_cfg.mean_col, sd_col=err_cfg.sd_col)

    set_seaborn_context(style)
    hue_order = compute_hue_order_by_metric(
        df_plot, metric_col="metric_name", target_metric=order_by_metric, mean_col=err_cfg.mean_col
    )

    fig, ax = draw_grouped_barplot(
        df=df_plot,
        metrics_order=metrics_order,
        metric_labels=metric_labels,
        palette=palette,
        hue_order=hue_order,
        style=style,
    )
    add_grouped_errorbars(
        ax=ax,
        df=df_plot,
        metrics_order=metrics_order,
        hue_order=hue_order,
        err_cfg=err_cfg,
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
    return fig, ax


def reformat_multigroup_df(df):
    '''Reformat df into ong form for plotting'''
    df_long = df.melt(
        id_vars=["measure", "set", "Name"],
        value_vars=[
            "test_best_model_acc", "test_best_model_f1", "test_best_model_mcc",
            "test_best_model_precision", "test_best_model_recall"
        ],
        var_name="Metric",
        value_name="Value"
    )
    
    # Pivot to get average and stdev as separate columns
    df_pivoted = df_long.pivot_table(
        index=["set", "Name", "Metric"],
        columns="measure",
        values="Value"
    ).reset_index()
    
    df_pivoted.columns.name = None
    df_pivoted.rename(columns={ "Name": "Split"}, inplace=True) #"average": "average", "stdev": "stdev",
    df_pivoted["Metric"] = df_pivoted["Metric"].str.replace("test_best_model_", "", regex=False).str.capitalize()
    
    return df_pivoted

DEFAULT_COLORS = {
    "1": "#FF0051", "1_ctrl": "#FF0051",
    "2": "#3C4860", "2_ctrl": "#3C4860",
    "3": "#444444", "3_ctrl": "#444444",
    "4": "#829CD0", "4_ctrl": "#829CD0",
    "5": "#5F7298", "5_ctrl": "#5F7298",
}

def plot_grouped_metric_barplot(
    df: pd.DataFrame,
    color_map: dict = None,
    *,
    metrics_order=("Accuracy", "F1 Score", "MCC", "Precision", "Recall"),
    metric_rename=None,
    ctrl_flag="ctrl",
    bar_width=0.10,
    group_spacing=0.50,
    figsize=(12, 6),
    ylabel="Mean ± SD (5-fold CV)",
    title="Model Performance Across Conditions by Metric",
    legend_title="Hold-out Set",
    rotate_xticks=30,
    save_path=None,
):
    """
    Draw grouped bars (mean ± SD) for each Metric, split by set and control status.

    Expects columns: ['Metric','set','Split','average','stdev'].
    - Rows whose 'Split' contains `ctrl_flag` (case-insensitive) are marked as control.
    - Colors are taken from `color_map` using group keys '<set>' or '<set>_ctrl'.
    """
    color_map = color_map or DEFAULT_COLORS
    metric_rename = metric_rename or {"Acc": "Accuracy", "F1": "F1 Score", "Mcc": "MCC"}

    needed = {"Metric", "set", "Split", "average", "stdev"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    data = df.copy()
    data["Metric"] = data["Metric"].replace(metric_rename)
    data = data[data["Metric"].isin(metrics_order)].copy()
    data["Metric"] = pd.Categorical(data["Metric"], categories=list(metrics_order), ordered=True)

    is_ctrl = data["Split"].str.contains(ctrl_flag, case=False, na=False)
    data["Group"] = data["set"].astype(str) + np.where(is_ctrl, "_ctrl", "")
    order_df = (
        data[["Group", "set"]].drop_duplicates().assign(
            set_num=lambda x: pd.to_numeric(x["set"], errors="coerce").fillna(0).astype(int),
            is_ctrl=data["Group"].str.endswith("_ctrl")
        ).sort_values(["set_num", "is_ctrl"], ascending=[True, True])
    )
    groups = order_df["Group"].tolist()
    n_groups = len(groups)
    group_width = n_groups * bar_width + group_spacing

    fig, ax = plt.subplots(figsize=figsize)
    x_ticks, x_labels = [], []

    for i, metric in enumerate(metrics_order):
        block = data[data["Metric"] == metric]
        if block.empty:
            continue
        base_x = i * group_width
        for j, g in enumerate(groups):
            row = block[block["Group"] == g]
            if row.empty:
                continue
            x = base_x + j * bar_width
            mean = float(row["average"].iloc[0])
            sd = float(row["stdev"].iloc[0])
            hatch = "//" if g.endswith("_ctrl") else ""
            ax.bar(
                x, mean, yerr=sd, width=bar_width,
                color=color_map.get(g, "#999999"),
                hatch=hatch, edgecolor="black", capsize=3,
                label=g if i == 0 else None,
            )
        x_center = base_x + (n_groups * bar_width) / 2 - bar_width / 2
        x_ticks.append(x_center); x_labels.append(metric)

    ax.set_xticks(x_ticks); ax.set_xticklabels(x_labels, rotation=rotate_xticks, ha="right")
    ax.set_ylabel(ylabel); ax.set_title(title)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_axisbelow(True); ax.yaxis.grid(True, color="#EEEEEE", linewidth=1)

    handles, labels = ax.get_legend_handles_labels()
    if labels:
        seen, H, L = set(), [], []
        for h, lab in zip(handles, labels):
            if lab not in seen:
                seen.add(lab); H.append(h); L.append(lab)
        ax.legend(H, L, title=legend_title, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig, ax



def plot_grouped_average_barplot(
    df: pd.DataFrame,
    *,
    metric_col: str = "Metric",
    split_col: str = "Split",   # your dataframe had "Split " accidentally; this function trims column names automatically
    mean_col: str = "average",
    sd_col: str = "stdev",
    metrics: Sequence[str] = ("Accuracy", "F1 Score", "MCC"),
    splits: Sequence[str] = (
        "random", "random_ctrl",
        "homology", "homology_ctrl",
        "organism", "organism_ctrl",
        "gene", "gene_ctrl",
        "antibiotic", "antibiotic_ctrl",
    ),
    fill_colors: Optional[Mapping[str, str]] = None,
    hatch_ctrl_substring: str = "ctrl",
    bar_width: float = 0.08,
    group_gap: float = 0.20,     # gap between metric groups
    figsize: Tuple[float, float] = (14, 6),
    ylabel: str = "Mean ± SD",
    title: Optional[str] = None,
    ylim: Tuple[float, float] = (-0.2, 1.05),
    save_path: Optional[str] = None,
):
    """
    Plot grouped bars (mean ± SD) for selected metrics, with each group showing all splits.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns [metric_col, split_col, mean_col, sd_col].
    metric_col, split_col, mean_col, sd_col : str
        Column names for metric, split label, mean, and standard deviation.
    metrics : sequence of str
        Metrics to show (order defines x-axis order).
    splits : sequence of str
        Split labels to show per metric (order defines bar order inside each group).
    fill_colors : mapping
        Map from split label to bar color. If None, a sensible default is used.
    hatch_ctrl_substring : str
        Bars whose split contains this substring (case-insensitive) are hatched.
    bar_width : float
        Width of each bar.
    group_gap : float
        Extra horizontal spacing between metric groups.
    figsize, ylabel, title, ylim : layout/label options.
    save_path : str
        If provided, save the figure here.

    Returns
    -------
    (fig, ax)
    """
    # Trim accidental whitespace in column names (e.g., "Split ")
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    # Defaults for colors (matches your original palette)
    if fill_colors is None:
        fill_colors = {
            "random": "#FF0051",       "random_ctrl": "#FF0051",
            "homology": "#3C4860",     "homology_ctrl": "#3C4860",
            "organism": "#EBEBEB",     "organism_ctrl": "#EBEBEB",
            "antibiotic": "#444444",   "antibiotic_ctrl": "#444444",
            "gene": "#829CD0",         "gene_ctrl": "#829CD0",
        }

    # Keep only requested metrics, in a stable (categorical) order
    df = df[df[metric_col].isin(metrics)].copy()
    df[metric_col] = pd.Categorical(df[metric_col], categories=list(metrics), ordered=True)

    # Precompute hatching (controls hatched)
    hatch_map = {s: ("//" if (hatch_ctrl_substring.lower() in s.lower()) else "") for s in splits}

    # Geometry: each metric group occupies n_splits * bar_width + group_gap
    n_splits = len(splits)
    group_width = n_splits * bar_width + group_gap

    fig, ax = plt.subplots(figsize=figsize)
    tick_positions, tick_labels = [], []

    for i, metric in enumerate(metrics):
        block = df[df[metric_col] == metric]
        if block.empty:
            continue

        base_x = i * group_width

        for j, split in enumerate(splits):
            row = block[block[split_col] == split]
            if row.empty:
                continue

            x = base_x + j * bar_width
            y = float(row[mean_col].iloc[0])
            yerr = float(row[sd_col].iloc[0])

            ax.bar(
                x, y, width=bar_width,
                color=fill_colors.get(split, "#999999"),
                edgecolor="black",
                hatch=hatch_map.get(split, ""),
                linewidth=1,
                label=split if i == 0 else None,  # add legend labels only once
            )
            ax.errorbar(x, y, yerr=yerr, fmt="none", ecolor="black", capsize=3, linewidth=1)

        # center tick under this metric's group
        center = base_x + (n_splits * bar_width) / 2 - (bar_width / 2)
        tick_positions.append(center)
        tick_labels.append(metric)

    # Cosmetics
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.set_ylim(*ylim)
    ax.tick_params(axis="both", labelsize=20)
    sns.despine()
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color="#EEEEEE", linewidth=1)

    # Legend dedupe
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        uniq = dict(zip(labels, handles))
        ax.legend(
            uniq.values(), uniq.keys(),
            bbox_to_anchor=(1.02, 1), loc="upper left",
            title="Split", fontsize=12, title_fontsize=12
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig, ax


def grouped_value_counts(
    df: pd.DataFrame,
    group_by: str,
    value_col: str,
    zero_label: str = '0',
    one_label: str = '1',
    prefix: str = 'count'
) -> pd.DataFrame:
    """
    Return a DataFrame giving counts of each value in `value_col`, 
    grouped by `group_by`.
    
    Parameters
    ----------
    df
        Your input DataFrame.
    group_by
        Column name to group on (e.g. 'gene', 'Organism').
    value_col
        Column name containing binary values (e.g. 'binary_predictions').
    zero_label, one_label
        How to name the 0/1 columns in the output.
    prefix
        Prefix for count columns, e.g. 'count' gives 'count_0' and 'count_1'.
        
    Returns
    -------
    pd.DataFrame
        One row per unique value of `group_by`, with columns:
        - `group_by`
        - `{prefix}_{zero_label}`, `{prefix}_{one_label}`
    """
    counts = (
        df
        .groupby([group_by, value_col])
        .size()
        .reset_index(name='count')
    )
    
    pivot = (
        counts
        .pivot(index=group_by, columns=value_col, values='count')
        .fillna(0)
        .astype(int)
    )
    
    rename_map = {}
    if 0 in pivot.columns:
        rename_map[0] = f"{prefix}_{zero_label}"
    if 1 in pivot.columns:
        rename_map[1] = f"{prefix}_{one_label}"
    pivot = pivot.rename(columns=rename_map)
    
    for lbl in (zero_label, one_label):
        colname = f"{prefix}_{lbl}"
        if colname not in pivot.columns:
            pivot[colname] = 0
    
    return pivot.reset_index()


def plot_stacked_dumbbell(
    df, group_col, actual_neg, pred_neg, actual_pos, pred_pos,
    color_an="#FF0051", color_pn="#3C4860",
    color_ap="#E69F00", color_pp="#56B4E9",
    offset=0.2, figsize=(8,12), font_size=12, alpha=0.8
):
    dfc = df.copy().sort_values(actual_pos, ascending=False)
    genes = dfc[group_col].tolist()
    y = list(range(len(genes)))
    y_neg = [yi - offset for yi in y]
    y_pos = [yi + offset for yi in y]

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(len(genes)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5,
                       color='lightgray', alpha=0.2, zorder=0)

    ax.scatter(dfc[actual_neg], y_neg, color=color_an, s=100, alpha=alpha, label="Actual Neg", zorder=3)
    ax.scatter(dfc[pred_neg],     y_neg, color=color_pn, s=100, alpha=alpha, label="Pred Neg",   zorder=3)
    for yi, row in zip(y_neg, dfc.itertuples()):
        ax.plot([getattr(row, actual_neg), getattr(row, pred_neg)],
                [yi, yi], color='gray', lw=1.5, zorder=2)

    ax.scatter(dfc[actual_pos], y_pos, color=color_ap, s=100, alpha=alpha, label="Actual Pos", zorder=3)
    ax.scatter(dfc[pred_pos],    y_pos, color=color_pp, s=100, alpha=alpha, label="Pred Pos",   zorder=3)
    for yi, row in zip(y_pos, dfc.itertuples()):
        ax.plot([getattr(row, actual_pos), getattr(row, pred_pos)],
                [yi, yi], color='gray', lw=1.5, zorder=2)

    ax.set_yticks(y)
    ax.set_yticklabels(genes, fontsize=font_size)
    ax.invert_yaxis()
    ax.set_xlabel("Count", fontsize=font_size+4)
    ax.set_ylabel(group_col, fontsize=font_size+4)
    #ax.set_title(f"{group_col} Neg vs Pos: Actual vs Predicted", fontsize=font_size+4, pad=15)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, linestyle=':', color='#CCCCCC')
    ax.tick_params(axis='x', labelsize=font_size)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
              frameon=False, fontsize=font_size,
              title_fontsize=font_size+2,
              bbox_to_anchor=(1.02,1), loc='upper left')

    plt.tight_layout()
    plt.savefig(f'{group_col}_dumbell.png', dpi=300)
    plt.show()

