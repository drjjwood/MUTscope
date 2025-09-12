import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import pandas as pd
import seaborn as sns

def plot_mean_reliability_curve(all_y_true, all_y_proba, n_bins=20):
    """
    Plot the mean reliability (calibration) curve across folds with ±1 SD band.

    Parameters
    ----------
    all_y_true : sequence of 1D arrays
        Ground-truth labels for each fold (0/1).
    all_y_proba : sequence of 1D arrays
        Predicted probabilities for each fold (same lengths as all_y_true).
    n_bins : int, default=20
        Number of calibration bins.
    strategy : {"uniform","quantile"}, default="uniform"
        Binning strategy passed to scikit-learn's `calibration_curve`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. If None, a new figure and axes are created.
    return_data : bool, default=False
        If True, also returns a dict with `bin_centers`, `mean_prob_true`, `std_prob_true`.

    Returns
    -------
    fig, ax, data (optional)
        Matplotlib figure/axes; and if requested, a dict with arrays for the plotted data.
    """
    if len(all_y_true) != len(all_y_proba):
        raise ValueError("all_y_true and all_y_proba must have the same length.")

    plt.figure(figsize=(6, 6))
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    prob_true_matrix = []

    for y_true, all_y_proba in zip(all_y_true, all_y_proba):
        prob_true, prob_pred = calibration_curve(
            y_true, all_y_proba, n_bins=n_bins, strategy='uniform'
        )
        prob_true_interp = np.interp(bin_centers, prob_pred, prob_true, left=np.nan, right=np.nan)
        prob_true_matrix.append(prob_true_interp)

    prob_true_matrix = np.array(prob_true_matrix)
    with np.errstate(invalid='ignore'):
        mean_prob_true = np.nanmean(prob_true_matrix, axis=0)
    std_prob_true = np.nanstd(prob_true_matrix, axis=0)
    plt.fill_between(bin_centers, mean_prob_true - std_prob_true, mean_prob_true + std_prob_true,
                     color='orange', alpha=0.2, label='±1 std')

    plt.plot(bin_centers, mean_prob_true, label='Mean calibration', color='#5F7298', linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Predicted Probability', fontsize=14)
    plt.ylabel('True Probability', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'calibration_curve.png', dpi=300)
    plt.show()

def expected_calibration_error(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 20,
    strategy: str = "uniform",
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum_i (n_i / N) * | mean(p_i) - frac_pos_i |

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground-truth labels (0/1).
    y_proba : array-like of shape (n_samples,)
        Predicted probabilities.
    n_bins : int, default=20
        Number of bins.
    strategy : {"uniform","quantile"}, default="uniform"
        Binning strategy. For 'uniform', bins are equally spaced in [0,1].
        For 'quantile', bins have (roughly) equal counts.

    Returns
    -------
    float
        Expected Calibration Error.
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    if y_true.shape[0] != y_proba.shape[0]:
        raise ValueError("y_true and y_proba must have the same length.")

    if strategy == "uniform":
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(y_proba, bins) - 1
    elif strategy == "quantile":
        # Use quantiles on probabilities to form bins with similar counts
        quantiles = np.linspace(0, 1, n_bins + 1)
        bins = np.quantile(y_proba, quantiles)
        # Handle potential duplicate edges (degenerate distributions)
        bins = np.unique(bins)
        # If unique bins < 2, fallback to uniform
        if bins.size < 2:
            bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(y_proba, bins, right=False) - 1
    else:
        raise ValueError("strategy must be 'uniform' or 'quantile'.")

    N = len(y_proba)
    ece = 0.0
    for i in range(n_bins):
        mask = bin_ids == i
        n_i = np.sum(mask)
        if n_i == 0:
            continue
        p_i = np.mean(y_proba[mask])
        frac_pos_i = np.mean(y_true[mask])  # fraction of positives in bin
        ece += (n_i / N) * abs(p_i - frac_pos_i)
    return float(ece)


def print_ece_brier_stats(all_y_true, all_y_proba, n_bins=20):
    """
    Print per-fold ECE and Brier score, plus means, and return arrays.

    Parameters
    ----------
    all_y_true : sequence of arrays
        Labels per fold.
    all_y_proba : sequence of arrays
        Probabilities per fold.
    n_bins : int, default=20
        Number of bins for ECE.
    strategy : {"uniform","quantile"}, default="uniform"
        Binning strategy for ECE.

    Returns
    -------
    eces, briers : np.ndarray, np.ndarray
        Per-fold ECEs and Brier scores.
    """
    eces = []
    briers = []
    for i in range(len(all_y_true)):
        y_true = all_y_true[i]
        y_proba = all_y_proba[i]
        
        ece = expected_calibration_error(y_true, y_proba, n_bins=n_bins)
        brier = brier_score_loss(y_true, y_proba)
        
        eces.append(ece)
        briers.append(brier)
        
        print(f"Fold {i} ECE: {ece:.4f}  |  Brier score: {brier:.4f}")
    
    print(f"\nMean ECE across folds:   {np.mean(eces):.4f}")
    print(f"Mean Brier score across folds: {np.mean(briers):.4f}")
    return eces, briers


def compute_risk_bins(y_true, y_prob, bins=[0.0, 0.1, 0.3, 0.6, 0.9, 1.0], verbose=True):
    """
    Maps predicted probabilities into calibrated risk categories.

    Args:
        y_true (np.ndarray): Ground truth labels (0/1).
        y_prob (np.ndarray): Predicted probabilities.
        bins (list): Bin edges to assign risk categories.
        verbose (bool): Print out bin stats.
    
    Returns:
        pd.DataFrame: DataFrame with bin ranges, mean predicted prob, observed rate, and risk label.
    """
    bin_labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    bin_ids = np.digitize(y_prob, bins) - 1

    risk_df = []

    for i in range(len(bins) - 1):
        mask = bin_ids == i
        n = np.sum(mask)
        if n == 0:
            continue
        bin_range = f"{bins[i]:.2f}–{bins[i+1]:.2f}"
        mean_pred = np.mean(y_prob[mask])
        tpr = np.mean(y_true[mask])
        label = bin_labels[i]
        risk_df.append({
            "Bin": bin_range,
            "Mean_N": np.round(n/5,0),
            "Mean_Pred_Prob": mean_pred,
            "Observed_Positive_Rate": tpr,
            "Risk_Label": label
        })

    df = pd.DataFrame(risk_df)
    if verbose:
        print(df)
    return df

def plot_risk_bins(risk_df):
    """
    Plot summary of risk bins as a horizontal bar chart.

    Args:
        risk_df (pd.DataFrame): Must contain columns ['Bin', 'N', 'Mean_Pred_Prob', 'Observed_Positive_Rate', 'Risk_Label']
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(risk_df['Risk_Label'], risk_df['Observed_Positive_Rate'], color='#5F7298')
    
    for bar, mean_prob in zip(bars, risk_df['Mean_Pred_Prob']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"Obs: {bar.get_width():.2f}\nPred: {mean_prob:.2f}",
                va='center', ha='left', fontsize=9)

    ax.set_xlabel("Observed Positive Rate", fontsize=14)
    plt.xlim(0, 1.05)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    sns.despine()
    plt.savefig(f'risk_bins.png', dpi=300)
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