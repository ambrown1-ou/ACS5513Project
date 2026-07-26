from base64 import b64encode
from io import BytesIO
from itertools import combinations

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
from sklearn.metrics import normalized_mutual_info_score

from .pipeline import FEATURE_FIELDS, load_training_data


def _encode_figure(figure):
    image = BytesIO()
    figure.savefig(image, format="png", dpi=120, bbox_inches="tight")
    return b64encode(image.getvalue()).decode("ascii")


def create_distribution_plot(data_path):
    """Return basic field distributions with summary statistics."""
    data = load_training_data(data_path)
    columns = 3
    rows = (len(FEATURE_FIELDS) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(14, rows * 4))
    axes = axes.flatten()
    try:
        for axis, field in zip(axes, FEATURE_FIELDS):
            values = data[field].dropna()
            avg = values.mean()
            med = values.median()
            
            if _is_discrete(values):
                # Basic stacked bar for categorical/encoded fields
                counts = data.groupby([field, "target"], observed=False).size().unstack(fill_value=0)
                # width=1.0 to eliminate space between bars
                counts.plot(kind="bar", stacked=True, color=["#b7d9c8", "#841617"], ax=axis, width=1.0, edgecolor="none")
                axis.set_xlabel("")
            else:
                # Basic histogram with no space between bars (bins touching)
                sns.histplot(data=data, x=field, hue="target", bins=15, multiple="stack", 
                             palette=["#b7d9c8", "#841617"], ax=axis, alpha=0.9, edgecolor="none", linewidth=0)
                axis.set_xlabel("")
            
            # Labeling and Styling
            axis.set_title(field.upper(), fontdict={'weight': 'bold', 'color': '#841617', 'size': 10})
            axis.grid(False)
            sns.despine(ax=axis, left=True) # Remove vertical axis line for simplicity
            axis.yaxis.set_visible(False)   # Hide Y axis labels for a cleaner "distribution only" look
            
            # Summary stats below the axis - centered and clear
            stats_text = f"AVG: {avg:.1f}  |  MED: {med:.0f}"
            axis.text(0.5, -0.25, stats_text, transform=axis.transAxes, 
                      ha='center', va='top', fontsize=9, color='#63736d',
                      fontweight='bold')
            
            if axis.legend_:
                axis.legend_.remove()
        
        for axis in axes[len(FEATURE_FIELDS):]:
            axis.remove()
        
        figure.suptitle("Clinical Feature Distributions & Benchmarks", fontsize=20, y=1.02, color="#841617", fontweight='bold')
        figure.patch.set_facecolor('#fdf9d8') # OU Cream background
        figure.tight_layout(pad=4.0)
        return _encode_figure(figure)
    finally:
        plt.close(figure)


def create_correlation_matrix_plot(data_path):
    """Return a heatmap showing feature-to-feature Pearson correlations."""
    data = load_training_data(data_path)
    figure, axis = plt.subplots(figsize=(11, 9))
    try:
        correlation = data[FEATURE_FIELDS].corr()
        sns.heatmap(
            correlation,
            ax=axis,
            cmap="RdYlBu_r",
            center=0,
            vmin=-1,
            vmax=1,
            annot=True,
            fmt=".2f",
            square=True,
            linewidths=0.4,
        )
        axis.set_title("Feature correlation matrix")
        figure.tight_layout()
        return _encode_figure(figure)
    finally:
        plt.close(figure)


def _is_discrete(series):
    return series.nunique() <= 10


def _normalized_mutual_information(first, second):
    """Measure general dependence after binning continuous values."""
    def discretize(series):
        if _is_discrete(series):
            return series.astype(str)
        return pd.qcut(series, q=10, labels=False, duplicates="drop").astype(str)

    return normalized_mutual_info_score(
        discretize(first), discretize(second), average_method="arithmetic"
    )


def create_strongest_correlations_plot(data_path, pair_count=4):
    """Return strongest feature pairs using normalized mutual information."""
    data = load_training_data(data_path)
    features = data[FEATURE_FIELDS]
    pearson = features.corr(method="pearson")
    spearman = features.corr(method="spearman")
    pairs = sorted(
        [
            (
                _normalized_mutual_information(features[first], features[second]),
                first,
                second,
            )
            for first, second in combinations(FEATURE_FIELDS, 2)
        ],
        reverse=True,
    )[:pair_count]
    figure, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    try:
        pair_details = []
        for axis, (mutual_information, first, second) in zip(axes, pairs):
            first_discrete = _is_discrete(features[first])
            second_discrete = _is_discrete(features[second])
            if first_discrete and second_discrete:
                counts = pd.crosstab(data[first], data[second])
                sns.heatmap(counts, annot=True, fmt="g", cmap="YlOrBr", ax=axis)
                axis.set_xlabel(second)
                axis.set_ylabel(first)
            elif first_discrete or second_discrete:
                x_field, y_field = (first, second) if first_discrete else (second, first)
                sns.boxplot(data=data, x=x_field, y=y_field, hue="target", ax=axis)
                axis.legend(title="Target", labels=["0", "1"])
                axis.set_xlabel(x_field)
                axis.set_ylabel(y_field)
            else:
                sns.scatterplot(
                    data=data,
                    x=first,
                    y=second,
                    hue="target",
                    palette="RdYlGn_r",
                    alpha=0.75,
                    ax=axis,
                    legend=False,
                )
                sns.regplot(
                    data=data,
                    x=first,
                    y=second,
                    scatter=False,
                    lowess=True,
                    line_kws={"color": "#e35b35", "linewidth": 2},
                    ax=axis,
                )
            pearson_value = pearson.loc[first, second]
            spearman_value = spearman.loc[first, second]
            axis.set_title(
                f"{first} vs {second}\n"
                f"NMI = {mutual_information:.2f}, Pearson r = {pearson_value:.2f}"
            )
            pair_details.append(
                {
                    "first": first,
                    "second": second,
                    "mutual_information": mutual_information,
                    "pearson": pearson_value,
                    "spearman": spearman_value,
                }
            )
        figure.suptitle("Strongest feature relationships by normalized mutual information", fontsize=16, y=1.01)
        figure.tight_layout()
        return _encode_figure(figure), pair_details
    finally:
        plt.close(figure)


def create_correlation_plot(data_path, fields, dimensions):
    """Return an interactive Plotly HTML 2D or 3D feature plot."""
    if dimensions not in (2, 3):
        raise ValueError("Choose either a 2D or 3D plot.")
    if len(fields) != dimensions or len(set(fields)) != len(fields):
        raise ValueError(f"Choose {dimensions} different parameters for this plot.")
    invalid_fields = [field for field in fields if field not in FEATURE_FIELDS]
    if invalid_fields:
        raise ValueError("Choose parameters from the available feature list.")

    data = load_training_data(data_path)
    
    if dimensions == 3:
        fig = px.scatter_3d(
            data, x=fields[0], y=fields[1], z=fields[2],
            color="target",
            color_continuous_scale="RdYlGn_r",
            opacity=0.8,
            title="Feature relationship by heart disease target"
        )
    else:
        fig = px.scatter(
            data, x=fields[0], y=fields[1],
            color="target",
            color_continuous_scale="RdYlGn_r",
            opacity=0.8,
            title="Feature relationship by heart disease target"
        )
    
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(title="Target (0=No, 1=Yes)")
    )
    
    return pio.to_html(fig, full_html=False, include_plotlyjs='cdn')
