from base64 import b64encode
from io import BytesIO
from itertools import combinations
from textwrap import dedent
import uuid

import matplotlib
import pandas as pd
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.metrics import normalized_mutual_info_score

from .pipeline import FEATURE_FIELDS, load_training_data


# Encodes a matplotlib figure as a base64 PNG string.
def _encode_figure(figure):
    image = BytesIO()
    figure.savefig(image, format="png", dpi=120, bbox_inches="tight")
    return b64encode(image.getvalue()).decode("ascii")


CATEGORICAL_COLOR_FIELDS = {
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal",
}


# Builds a bar chart for the main training metrics of one model run.
def create_training_metrics_plot(metrics, title=None):
    metric_names = ["accuracy", "precision", "recall", "f1"]
    labels = [name.upper() if name != "f1" else "F1" for name in metric_names]
    values = [float(metrics.get(name, 0)) * 100 for name in metric_names]
    colors = ["#841617", "#b23a3d", "#d67d7f", "#b7d9c8"]

    figure, axis = plt.subplots(figsize=(8, 5))
    try:
        bars = axis.bar(labels, values, color=colors, width=0.6)
        axis.set_ylim(0, 100)
        axis.set_ylabel("Score (%)")
        axis.set_title(title or "Training Metrics")
        axis.grid(axis="y", alpha=0.2)
        axis.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3)
        figure.tight_layout()
        return _encode_figure(figure)
    finally:
        plt.close(figure)


# Builds a horizontal bar chart showing the relative importance of each feature.
def create_feature_importance_plot(importances, title=None):
    if not importances:
        return None

    ordered_items = sorted(importances.items(), key=lambda item: item[1], reverse=True)
    labels = [str(name).replace("_", " ").title() for name, _ in ordered_items]
    values = [float(value) * 100 for _, value in ordered_items]

    figure, axis = plt.subplots(figsize=(9, max(4.5, 0.35 * len(labels) + 1.5)))
    try:
        bars = axis.barh(labels, values, color="#841617")
        axis.invert_yaxis()
        axis.set_xlabel("Relative importance (%)")
        axis.set_title(title or "Feature Importance")
        axis.grid(axis="x", alpha=0.2)
        axis.bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=3)
        figure.tight_layout()
        return _encode_figure(figure)
    finally:
        plt.close(figure)
# Normalizes the selected color field for grouped or continuous Plotly coloring.
def _prepare_color_data(data, color_field):
    plot_data = data.copy()
    if color_field in CATEGORICAL_COLOR_FIELDS:
        category_order = sorted(plot_data[color_field].dropna().unique())
        plot_data[color_field] = plot_data[color_field].astype(str)
        return plot_data, True, [str(value) for value in category_order]
    return plot_data, False, None


def _prepare_plot_data(data, missing_strategy):
    if missing_strategy != "impute":
        return data

    plot_data = data.copy()
    for field in FEATURE_FIELDS:
        median = plot_data[field].median()
        if pd.notna(median):
            plot_data[field] = plot_data[field].fillna(median)
    return plot_data


def _axis_range(series, padding=0.05):
        values = pd.to_numeric(series, errors="coerce").dropna()
        if values.empty:
                return [0.0, 1.0]

        minimum = float(values.min())
        maximum = float(values.max())
        if minimum == maximum:
                delta = abs(minimum) if minimum else 1.0
                minimum -= delta * 0.5
                maximum += delta * 0.5

        span = maximum - minimum
        margin = span * padding
        return [minimum - margin, maximum + margin]


def _fit_line(data, x_field, y_field):
        subset = data[[x_field, y_field]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(subset) < 2 or subset[x_field].nunique() < 2:
                return None

        x_values = subset[x_field].to_numpy(dtype=float)
        y_values = subset[y_field].to_numpy(dtype=float)
        design = np.column_stack([x_values, np.ones(len(x_values))])
        slope, intercept = np.linalg.lstsq(design, y_values, rcond=None)[0]
        return float(slope), float(intercept)


def _fit_plane(data, x_field, y_field, z_field):
        subset = data[[x_field, y_field, z_field]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(subset) < 3:
                return None

        x_values = subset[x_field].to_numpy(dtype=float)
        y_values = subset[y_field].to_numpy(dtype=float)
        z_values = subset[z_field].to_numpy(dtype=float)
        design = np.column_stack([x_values, y_values, np.ones(len(x_values))])
        coefficients = np.linalg.lstsq(design, z_values, rcond=None)[0]
        return [float(value) for value in coefficients]


def _build_2d_regression_post_script(plot_id, regression_index, x_range):
        script = dedent(
                """
                (function () {
                    const plotId = '__PLOT_ID__';
                    const regressionIndex = __REGRESSION_INDEX__;
                    const xRange = __X_RANGE__;

                    function collectVisiblePairs(gd) {
                        const pairs = [];
                        gd.data.forEach((trace) => {
                            if (!trace || trace.visible === false || trace.visible === 'legendonly') {
                                return;
                            }
                            if (trace.meta && trace.meta.role === 'regression') {
                                return;
                            }
                            if (!Array.isArray(trace.x) || !Array.isArray(trace.y)) {
                                return;
                            }

                            const limit = Math.min(trace.x.length, trace.y.length);
                            for (let index = 0; index < limit; index += 1) {
                                const xValue = Number(trace.x[index]);
                                const yValue = Number(trace.y[index]);
                                if (Number.isFinite(xValue) && Number.isFinite(yValue)) {
                                    pairs.push([xValue, yValue]);
                                }
                            }
                        });
                        return pairs;
                    }

                    function fitLine(pairs) {
                        if (pairs.length < 2) {
                            return null;
                        }

                        let sumX = 0;
                        let sumY = 0;
                        let sumXX = 0;
                        let sumXY = 0;

                        pairs.forEach(([xValue, yValue]) => {
                            sumX += xValue;
                            sumY += yValue;
                            sumXX += xValue * xValue;
                            sumXY += xValue * yValue;
                        });

                        const count = pairs.length;
                        const denominator = (count * sumXX) - (sumX * sumX);
                        if (Math.abs(denominator) < 1e-12) {
                            return null;
                        }

                        const slope = ((count * sumXY) - (sumX * sumY)) / denominator;
                        const intercept = (sumY - (slope * sumX)) / count;
                        return { slope, intercept };
                    }

                    function updateRegression() {
                        const gd = document.getElementById(plotId);
                        if (!gd || !gd.data) {
                            return;
                        }

                        const fit = fitLine(collectVisiblePairs(gd));
                        if (!fit) {
                            Plotly.restyle(gd, { visible: [false] }, [regressionIndex]);
                            return;
                        }

                        const lineX = [xRange[0], xRange[1]];
                        const lineY = lineX.map((value) => (fit.slope * value) + fit.intercept);
                        Plotly.restyle(gd, { x: [lineX], y: [lineY], visible: [true] }, [regressionIndex]);
                    }

                    function bind() {
                        const gd = document.getElementById(plotId);
                        if (!gd || gd.dataset.regressionBound === 'true') {
                            return;
                        }

                        gd.dataset.regressionBound = 'true';
                        gd.on('plotly_restyle', function (eventData) {
                            const indices = Array.isArray(eventData) ? eventData[1] : null;
                            if (Array.isArray(indices) && indices.length === 1 && indices[0] === regressionIndex) {
                                return;
                            }
                            updateRegression();
                        });
                        gd.on('plotly_legendclick', function () {
                            window.setTimeout(updateRegression, 0);
                        });
                        gd.on('plotly_legenddoubleclick', function () {
                            window.setTimeout(updateRegression, 0);
                        });
                        updateRegression();
                    }

                    if (document.readyState === 'loading') {
                        document.addEventListener('DOMContentLoaded', bind);
                    } else {
                        bind();
                    }
                }());
                """
        )
        return (
                script.replace("__PLOT_ID__", plot_id)
                .replace("__REGRESSION_INDEX__", str(regression_index))
                .replace("__X_RANGE__", repr(list(x_range)))
        )


def _build_3d_regression_post_script(plot_id, regression_index, x_range, y_range):
        script = dedent(
                """
                (function () {
                    const plotId = '__PLOT_ID__';
                    const regressionIndex = __REGRESSION_INDEX__;
                    const xRange = __X_RANGE__;
                    const yRange = __Y_RANGE__;

                    function collectVisiblePoints(gd) {
                        const points = [];
                        gd.data.forEach((trace) => {
                            if (!trace || trace.visible === false || trace.visible === 'legendonly') {
                                return;
                            }
                            if (trace.meta && trace.meta.role === 'regression') {
                                return;
                            }
                            if (!Array.isArray(trace.x) || !Array.isArray(trace.y) || !Array.isArray(trace.z)) {
                                return;
                            }

                            const limit = Math.min(trace.x.length, trace.y.length, trace.z.length);
                            for (let index = 0; index < limit; index += 1) {
                                const xValue = Number(trace.x[index]);
                                const yValue = Number(trace.y[index]);
                                const zValue = Number(trace.z[index]);
                                if (Number.isFinite(xValue) && Number.isFinite(yValue) && Number.isFinite(zValue)) {
                                    points.push([xValue, yValue, zValue]);
                                }
                            }
                        });
                        return points;
                    }

                    function solve3x3(matrix, vector) {
                        const augmented = matrix.map((row, index) => row.concat([vector[index]]));

                        for (let pivot = 0; pivot < 3; pivot += 1) {
                            let maxRow = pivot;
                            for (let row = pivot + 1; row < 3; row += 1) {
                                if (Math.abs(augmented[row][pivot]) > Math.abs(augmented[maxRow][pivot])) {
                                    maxRow = row;
                                }
                            }

                            if (Math.abs(augmented[maxRow][pivot]) < 1e-12) {
                                return null;
                            }

                            if (maxRow !== pivot) {
                                const temp = augmented[pivot];
                                augmented[pivot] = augmented[maxRow];
                                augmented[maxRow] = temp;
                            }

                            const pivotValue = augmented[pivot][pivot];
                            for (let column = pivot; column < 4; column += 1) {
                                augmented[pivot][column] /= pivotValue;
                            }

                            for (let row = 0; row < 3; row += 1) {
                                if (row === pivot) {
                                    continue;
                                }

                                const factor = augmented[row][pivot];
                                for (let column = pivot; column < 4; column += 1) {
                                    augmented[row][column] -= factor * augmented[pivot][column];
                                }
                            }
                        }

                        return [augmented[0][3], augmented[1][3], augmented[2][3]];
                    }

                    function fitPlane(points) {
                        if (points.length < 3) {
                            return null;
                        }

                        let sumXX = 0;
                        let sumXY = 0;
                        let sumYY = 0;
                        let sumX = 0;
                        let sumY = 0;
                        let sumXZ = 0;
                        let sumYZ = 0;
                        let sumZ = 0;

                        points.forEach(([xValue, yValue, zValue]) => {
                            sumXX += xValue * xValue;
                            sumXY += xValue * yValue;
                            sumYY += yValue * yValue;
                            sumX += xValue;
                            sumY += yValue;
                            sumXZ += xValue * zValue;
                            sumYZ += yValue * zValue;
                            sumZ += zValue;
                        });

                        const matrix = [
                            [sumXX, sumXY, sumX],
                            [sumXY, sumYY, sumY],
                            [sumX, sumY, points.length],
                        ];
                        const vector = [sumXZ, sumYZ, sumZ];
                        return solve3x3(matrix, vector);
                    }

                    function buildGrid(coefficients) {
                        const xValues = Array.from({ length: 20 }, (_, index) => xRange[0] + ((xRange[1] - xRange[0]) * index) / 19);
                        const yValues = Array.from({ length: 20 }, (_, index) => yRange[0] + ((yRange[1] - yRange[0]) * index) / 19);
                        const zValues = yValues.map((yValue) => xValues.map((xValue) => (coefficients[0] * xValue) + (coefficients[1] * yValue) + coefficients[2]));
                        return { xValues, yValues, zValues };
                    }

                    function updateRegression() {
                        const gd = document.getElementById(plotId);
                        if (!gd || !gd.data) {
                            return;
                        }

                        const fit = fitPlane(collectVisiblePoints(gd));
                        if (!fit) {
                            Plotly.restyle(gd, { visible: [false] }, [regressionIndex]);
                            return;
                        }

                        const grid = buildGrid(fit);
                        Plotly.restyle(
                            gd,
                            { x: [grid.xValues], y: [grid.yValues], z: [grid.zValues], visible: [true] },
                            [regressionIndex]
                        );
                    }

                    function bind() {
                        const gd = document.getElementById(plotId);
                        if (!gd || gd.dataset.regressionBound === 'true') {
                            return;
                        }

                        gd.dataset.regressionBound = 'true';
                        gd.on('plotly_restyle', function (eventData) {
                            const indices = Array.isArray(eventData) ? eventData[1] : null;
                            if (Array.isArray(indices) && indices.length === 1 && indices[0] === regressionIndex) {
                                return;
                            }
                            updateRegression();
                        });
                        gd.on('plotly_legendclick', function () {
                            window.setTimeout(updateRegression, 0);
                        });
                        gd.on('plotly_legenddoubleclick', function () {
                            window.setTimeout(updateRegression, 0);
                        });
                        updateRegression();
                    }

                    if (document.readyState === 'loading') {
                        document.addEventListener('DOMContentLoaded', bind);
                    } else {
                        bind();
                    }
                }());
                """
        )
        return (
                script.replace("__PLOT_ID__", plot_id)
                .replace("__REGRESSION_INDEX__", str(regression_index))
                .replace("__X_RANGE__", repr(list(x_range)))
                .replace("__Y_RANGE__", repr(list(y_range)))
        )


# Builds an interactive Plotly 2D or 3D feature plot with selectable color encoding.
def create_correlation_plot(data_path, fields, dimensions, missing_strategy="impute"):
    if dimensions not in (2, 3):
        raise ValueError("Choose either a 2D or 3D plot.")

    expected_fields = dimensions + 1
    if len(fields) != expected_fields or len(set(fields)) != len(fields):
        raise ValueError(f"Choose {expected_fields} different parameters for this plot.")

    invalid_fields = [field for field in fields if field not in FEATURE_FIELDS]
    if invalid_fields:
        raise ValueError("Choose parameters from the available feature list.")

    # Load data according to requested missing-data handling so the UI can
    # toggle between dropping incomplete rows or preserving them for plots.
    data = _prepare_plot_data(
        load_training_data(data_path, missing_strategy=missing_strategy),
        missing_strategy,
    )
    axis_fields = fields[:dimensions]
    color_field = fields[dimensions]
    x_range = _axis_range(data[axis_fields[0]])
    y_range = _axis_range(data[axis_fields[1]])
    z_range = _axis_range(data[axis_fields[2]]) if dimensions == 3 else None
    plot_data, grouped_color, category_order = _prepare_color_data(data, color_field)
    display_color_field = color_field.replace("_", " ").title()

    plot_kwargs = {
        "opacity": 0.8,
        "title": f"{dimensions}D feature relationship colored by {display_color_field}",
    }
    if grouped_color:
        plot_kwargs["category_orders"] = {color_field: category_order}
        plot_kwargs["color_discrete_sequence"] = [
            "#841617",
            "#b23a3d",
            "#d67d7f",
            "#b7d9c8",
            "#63736d",
            "#f0c808",
            "#5c7cfa",
            "#2a9d8f",
        ]
    else:
        plot_kwargs["color_continuous_scale"] = "RdYlGn_r"

    plot_id = f"feature-plot-{uuid.uuid4().hex}"
    if dimensions == 3:
        fig = px.scatter_3d(
            plot_data,
            x=axis_fields[0],
            y=axis_fields[1],
            z=axis_fields[2],
            color=color_field,
            **plot_kwargs,
        )

        coefficients = _fit_plane(data, axis_fields[0], axis_fields[1], axis_fields[2])
        if coefficients:
            grid_x = np.linspace(x_range[0], x_range[1], 20)
            grid_y = np.linspace(y_range[0], y_range[1], 20)
            grid_x_mesh, grid_y_mesh = np.meshgrid(grid_x, grid_y)
            grid_z_mesh = (
                coefficients[0] * grid_x_mesh
                + coefficients[1] * grid_y_mesh
                + coefficients[2]
            )
            fig.add_trace(
                go.Surface(
                    x=grid_x,
                    y=grid_y,
                    z=grid_z_mesh,
                    name="Linear regression plane",
                    showscale=False,
                    opacity=0.35,
                    hoverinfo="skip",
                    colorscale=[[0, "#841617"], [1, "#d67d7f"]],
                    showlegend=False,
                    meta={"role": "regression"},
                )
            )
        else:
            fig.add_trace(
                go.Surface(
                    x=[0, 1],
                    y=[0, 1],
                    z=[[0, 0], [0, 0]],
                    name="Linear regression plane",
                    showscale=False,
                    opacity=0.35,
                    hoverinfo="skip",
                    colorscale=[[0, "#841617"], [1, "#d67d7d"]],
                    showlegend=False,
                    visible=False,
                    meta={"role": "regression"},
                )
            )
    else:
        fig = px.scatter(
            plot_data,
            x=axis_fields[0],
            y=axis_fields[1],
            color=color_field,
            **plot_kwargs,
        )

        coefficients = _fit_line(data, axis_fields[0], axis_fields[1])
        if coefficients:
            line_x = np.linspace(x_range[0], x_range[1], 2)
            line_y = (coefficients[0] * line_x) + coefficients[1]
            fig.add_trace(
                go.Scatter(
                    x=line_x,
                    y=line_y,
                    mode="lines",
                    name="Linear regression",
                    line=dict(color="#841617", width=3),
                    hoverinfo="skip",
                    showlegend=False,
                    meta={"role": "regression"},
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="Linear regression",
                    line=dict(color="#841617", width=3),
                    hoverinfo="skip",
                    showlegend=False,
                    visible=False,
                    meta={"role": "regression"},
                )
            )

    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40))
    fig.update_layout(uirevision="feature-regression")
    if dimensions == 3:
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=x_range, autorange=False),
                yaxis=dict(range=y_range, autorange=False),
                zaxis=dict(range=z_range, autorange=False),
            )
        )
    else:
        fig.update_xaxes(range=x_range, autorange=False)
        fig.update_yaxes(range=y_range, autorange=False)

    if grouped_color:
        fig.update_layout(legend=dict(title=display_color_field))
    else:
        fig.update_layout(coloraxis_colorbar=dict(title=display_color_field))

    post_script = (
        _build_3d_regression_post_script(plot_id, len(fig.data) - 1, x_range, y_range)
        if dimensions == 3
        else _build_2d_regression_post_script(plot_id, len(fig.data) - 1, x_range)
    )
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs='cdn',
        post_script=post_script,
        div_id=plot_id,
        config={"responsive": True},
    )


def create_distribution_plot(data_path, missing_strategy="impute"):
    """Return basic field distributions with summary statistics.

    Accepts `missing_strategy` to control whether incomplete rows are dropped
    or preserved for visualization (e.g. 'drop' or 'impute').
    """
    data = _prepare_plot_data(
        load_training_data(data_path, missing_strategy=missing_strategy),
        missing_strategy,
    )
    diagnosis_labels = ["No Disease", "Heart Disease"]
    diagnosis_colors = {
        "No Disease": "#b7d9c8",
        "Heart Disease": "#841617",
    }
    plot_data = data.assign(
        diagnosis=data["target"].map({0: diagnosis_labels[0], 1: diagnosis_labels[1]})
    )
    columns = 3
    rows = (len(FEATURE_FIELDS) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(14, rows * 4))
    axes = axes.flatten()
    legend_handles = None
    legend_labels = None
    try:
        for axis, field in zip(axes, FEATURE_FIELDS):
            values = data[field].dropna()
            avg = values.mean()
            med = values.median()

            if _is_discrete(values):
                counts = (
                    plot_data.groupby([field, "diagnosis"], observed=False)
                    .size()
                    .unstack(fill_value=0)
                    .reindex(columns=diagnosis_labels, fill_value=0)
                )
                counts.plot(
                    kind="bar",
                    stacked=False,
                    color=[diagnosis_colors[label] for label in diagnosis_labels],
                    ax=axis,
                    width=0.82,
                    edgecolor="white",
                    linewidth=0.4,
                )
            else:
                sns.histplot(
                    data=plot_data,
                    x=field,
                    hue="diagnosis",
                    hue_order=diagnosis_labels,
                    bins=15,
                    multiple="dodge",
                    shrink=0.86,
                    common_bins=True,
                    palette=diagnosis_colors,
                    ax=axis,
                    alpha=0.9,
                    edgecolor="white",
                    linewidth=0.4,
                )

            # Labeling and Styling
            axis.set_title(field.upper(), fontdict={'weight': 'bold', 'color': '#841617', 'size': 10})
            axis.grid(False)
            axis.set_xlabel(field.replace("_", " ").title())
            axis.set_ylabel("Patients")
            axis.tick_params(axis="y", labelsize=8)
            sns.despine(ax=axis, left=False)

            # Summary stats below the axis - centered and clear
            stats_text = f"AVG: {avg:.1f}  |  MED: {med:.1f}"
            axis.text(0.5, -0.25, stats_text, transform=axis.transAxes,
                      ha='center', va='top', fontsize=9, color='#63736d',
                      fontweight='bold')

            if legend_handles is None:
                legend_handles, legend_labels = axis.get_legend_handles_labels()
            if axis.legend_:
                axis.legend_.remove()

        for axis in axes[len(FEATURE_FIELDS):]:
            axis.remove()

        if legend_handles:
            figure.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.985),
                ncol=2,
                frameon=False,
            )
        figure.suptitle("Clinical Feature Distributions & Benchmarks", fontsize=20, y=1.04, color="#841617", fontweight='bold')
        figure.patch.set_facecolor('#fdf9d8')
        figure.tight_layout(rect=(0, 0, 1, 0.92), pad=2.5)
        return _encode_figure(figure)
    finally:
        plt.close(figure)
def create_correlation_matrix_plot(data_path, missing_strategy="impute"):
    """Return a heatmap showing feature-to-feature Pearson correlations.

    The `missing_strategy` argument controls how missing feature values are
    handled before computing correlations.
    """
    data = _prepare_plot_data(
        load_training_data(data_path, missing_strategy=missing_strategy),
        missing_strategy,
    )
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


# Returns the strongest feature pairs by normalized mutual information.
def create_strongest_correlations_plot(data_path, pair_count=4, missing_strategy="impute"):
    """Return strongest feature pairs using normalized mutual information.

    Accepts `missing_strategy` to control handling of missing values for the
    exploratory computation.
    """
    data = _prepare_plot_data(
        load_training_data(data_path, missing_strategy=missing_strategy),
        missing_strategy,
    )
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
