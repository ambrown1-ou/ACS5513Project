import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config import paths as project_paths
from model import FEATURE_FIELDS, load_training_data


DATASET_PATH = project_paths.BUNDLED_DATASETS_DIR / "heart_disease_cleveland_cleaned.csv"
CV_FOLDS = 5

models = {
    "KNN_K5": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=5),
    ),
    "Naive Bayes": make_pipeline(
        SimpleImputer(strategy="median"),
        GaussianNB(),
    ),
    "SVM": make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        SVC(kernel="rbf"),
    )
}

# models = {}

# for k in range(0, 20):
#     if k == 0:
#         continue
#     if f"KNN_K{str(k).rjust(2, '0')}" not in models:
#         models[f"KNN_K{str(k).rjust(2, '0')}"] = make_pipeline(
#             SimpleImputer(strategy="median"),
#             StandardScaler(),
#             KNeighborsClassifier(n_neighbors=k),
#         )



# Add logistic regression to the model comparison
# add 
def compare_models(
    data_path,
    cv_folds=5,
    random_state=42,
    output_dir=project_paths.GRAPHICS_DIR,
):
    data = load_training_data(
        data_path,
        feature_fields=FEATURE_FIELDS,
        missing_strategy="impute",
    )
    features = data[FEATURE_FIELDS].astype(float)
    target = data["target"].astype(int)

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    scoring = {
        "Accuracy": "accuracy",
        "Precision": make_scorer(precision_score, zero_division=0),
        "Recall": make_scorer(recall_score, zero_division=0),
        "F1": make_scorer(f1_score, zero_division=0),
    }

    results = []
    for name, model in models.items():
        scores = cross_validate(
            model,
            features,
            target,
            cv=cv,
            scoring=scoring,
        )
        results.append(
            {
                "Model": name,
                "Accuracy": scores["test_Accuracy"].mean(),
                "Accuracy Std": scores["test_Accuracy"].std(),
                "Precision": scores["test_Precision"].mean(),
                "Precision Std": scores["test_Precision"].std(),
                "Recall": scores["test_Recall"].mean(),
                "Recall Std": scores["test_Recall"].std(),
                "F1": scores["test_F1"].mean(),
                "F1 Std": scores["test_F1"].std(),
            }
        )

    results_df = pd.DataFrame(results)
    visualize_results(results_df, output_dir=output_dir)
    return results_df.sort_values("F1", ascending=False).reset_index(drop=True), len(data)


def run_naive_bayes_feature_ablation(
    features,
    target,
    cv_folds=5,
    random_state=42,
):
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )
    scoring = {
        "Accuracy": "accuracy",
        "Precision": make_scorer(precision_score, zero_division=0),
        "Recall": make_scorer(recall_score, zero_division=0),
        "F1": make_scorer(f1_score, zero_division=0),
    }

    naive_bayes_model = make_pipeline(
        SimpleImputer(strategy="median"),
        GaussianNB(),
    )

    baseline_scores = cross_validate(
        naive_bayes_model,
        features,
        target,
        cv=cv,
        scoring=scoring,
    )
    baseline_means = {
        metric: baseline_scores[f"test_{metric}"].mean() for metric in scoring
    }

    rows = []
    for feature_name in features.columns:
        reduced_features = features.drop(columns=[feature_name])
        reduced_scores = cross_validate(
            naive_bayes_model,
            reduced_features,
            target,
            cv=cv,
            scoring=scoring,
        )
        reduced_means = {
            metric: reduced_scores[f"test_{metric}"].mean() for metric in scoring
        }
        rows.append(
            {
                "Removed Feature": feature_name,
                "Baseline Accuracy": baseline_means["Accuracy"],
                "Ablated Accuracy": reduced_means["Accuracy"],
                "Delta Accuracy": baseline_means["Accuracy"] - reduced_means["Accuracy"],
                "Baseline Precision": baseline_means["Precision"],
                "Ablated Precision": reduced_means["Precision"],
                "Delta Precision": baseline_means["Precision"] - reduced_means["Precision"],
                "Baseline Recall": baseline_means["Recall"],
                "Ablated Recall": reduced_means["Recall"],
                "Delta Recall": baseline_means["Recall"] - reduced_means["Recall"],
                "Baseline F1": baseline_means["F1"],
                "Ablated F1": reduced_means["F1"],
                "Delta F1": baseline_means["F1"] - reduced_means["F1"],
            }
        )

    return pd.DataFrame(rows).sort_values("Delta F1", ascending=False).reset_index(drop=True)

# Compare models visually by plotting each parameter's mean score with error bars.
# Each graph will have the model names on the x-axis and the mean score on the y-axis, 
# with error bars representing the standard deviation of the scores when available.
def visualize_results(results_df, output_dir=project_paths.GRAPHICS_DIR):
    import seaborn as sns
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files = {}

    for metric in ["Accuracy", "Precision", "Recall", "F1"]:
        figure, axis = plt.subplots(figsize=(8, 6))
        sns.barplot(
            x="Model",
            y=metric,
            data=results_df,
            ci=None,
            palette="muted",
            ax=axis,
        )
        standard_deviation = results_df.get(f"{metric} Std")
        if standard_deviation is not None:
            error_values = pd.to_numeric(standard_deviation, errors="coerce")
            if error_values.notna().any():
                axis.errorbar(
                    x=range(len(results_df)),
                    y=results_df[metric],
                    yerr=error_values.fillna(0),
                    fmt="none",
                    c="black",
                    capsize=5,
                )
        axis.set_title(f"Model Comparison: {metric}")
        axis.set_ylabel(metric)
        axis.set_xlabel("Model")
        axis.set_ylim(0, 1)
        axis.grid(axis="y")
        figure.tight_layout()
        output_path = output_dir / f"model_{metric.lower()}_comparison.png"
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        saved_files[metric] = output_path
        plt.show()
        plt.close(figure)

    return saved_files

# graph k only models as line graph, k on x axis, metric on y axis, with error bars representing the standard deviation of the scores when available.

def graph_k_only(results_df, output_dir=project_paths.GRAPHICS_DIR):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    model_names = results_df["Model"].astype(str)
    knn_results = results_df.loc[model_names.str.fullmatch(r"KNN_K\d+")].copy()
    if knn_results.empty:
        raise ValueError("No KNN results found. Expected model names such as 'KNN_K5'.")

    knn_results["K"] = (
        knn_results["Model"]
        .str.extract(r"KNN_K(?P<neighbors>\d+)", expand=False)
        .astype(int)
    )
    if knn_results["K"].duplicated().any():
        raise ValueError("The validation results contain duplicate K values.")
    knn_results = knn_results.sort_values("K").reset_index(drop=True)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_maxima = {}
    saved_files = {}
    maxima_rows = []
    all_k_values = knn_results["K"].tolist()

    y_limits = {}
    for metric in ["Accuracy", "Precision", "Recall", "F1"]:
        values = knn_results[metric].astype(float)
        standard_deviation = knn_results.get(f"{metric} Std")
        error_values = (
            standard_deviation.fillna(0).astype(float)
            if standard_deviation is not None
            else pd.Series(0.0, index=knn_results.index)
        )
        lower_bound = float((values - error_values).min())
        upper_bound = float((values + error_values).max())
        padding = max((upper_bound - lower_bound) * 0.1, 0.01)
        y_limits[metric] = (
            max(0.0, float(lower_bound - padding)),
            min(1.0, float(upper_bound + padding)),
        )

    for metric in ["Accuracy", "Precision", "Recall", "F1"]:
        values = knn_results[metric].astype(float)
        standard_deviation = knn_results.get(f"{metric} Std")
        error_values = (
            standard_deviation.fillna(0).astype(float)
            if standard_deviation is not None
            else pd.Series(0.0, index=knn_results.index)
        )
        local_maxima_mask = values.gt(values.shift(1)) & values.gt(values.shift(-1))
        metric_maxima = knn_results.loc[local_maxima_mask, ["K", metric]]
        local_maxima[metric] = [
            {"K": int(row["K"]), "value": float(row[metric])}
            for _, row in metric_maxima.iterrows()
        ]
        maxima_rows.extend(
            {
                "Metric": metric,
                "K": int(row["K"]),
                "Value": float(row[metric]),
            }
            for _, row in metric_maxima.iterrows()
        )

        figure, axis = plt.subplots(figsize=(10, 6))
        axis.plot(knn_results["K"], values, marker="o")
        if standard_deviation is not None:
            axis.errorbar(
                knn_results["K"],
                values,
                yerr=error_values,
                fmt="none",
                c="black",
                capsize=5,
            )

        for _, maximum in metric_maxima.iterrows():
            axis.scatter(maximum["K"], maximum[metric], color="crimson", zorder=3)
            axis.annotate(
                f"K={int(maximum['K'])}\n{maximum[metric]:.2%}",
                xy=(maximum["K"], maximum[metric]),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                color="crimson",
            )

        axis.set_title(f"KNN Model Comparison: {metric} ({CV_FOLDS}-fold validation)")
        axis.set_ylabel(metric)
        axis.set_xlabel("K")
        axis.set_xticks(all_k_values)
        axis.set_xlim(min(all_k_values) - 0.5, max(all_k_values) + 0.5)
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
        axis.set_ylim(*y_limits[metric])
        axis.grid(axis="y")
        figure.tight_layout()
        output_path = output_dir / f"knn_{metric.lower()}_by_k.png"
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        saved_files[metric] = output_path
        plt.show()
        plt.close(figure)

    maxima_path = output_dir / "knn_local_maxima.csv"
    pd.DataFrame(
        maxima_rows,
        columns=["Metric", "K", "Value"],
    ).to_csv(maxima_path, index=False)
    saved_files["local_maxima"] = maxima_path
    return {"local_maxima": local_maxima, "files": saved_files, "y_limits": y_limits}

def main():
    parser = argparse.ArgumentParser(
        description="Compare KNN, Naive Bayes, and SVM on a heart-disease dataset."
    )
    parser.add_argument(
        "data_path",
        nargs="?",
        type=Path,
        default=DATASET_PATH,
        help="CSV file containing the canonical feature columns and target.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_paths.GRAPHICS_DIR,
        help="Directory where comparison plots and peak data are saved.",
    )
    args = parser.parse_args()


    results, record_count = compare_models(
        args.data_path,
        cv_folds=CV_FOLDS,
        random_state=args.random_state,
            output_dir=args.output_dir,
    )
    results_for_print = results.sort_values("F1", ascending=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "knn_model_comparison.csv"
    results.to_csv(results_path, index=False)
    graph_report = None
    ablation_results = None

    try:
        if all("KNN" in model_name for model_name in results["Model"]):
            graph_report = graph_k_only(results, output_dir=args.output_dir)

    except Exception as e:
        print(f"Error graphing KNN results: {e}")

    try:
        source_data = load_training_data(
            args.data_path,
            feature_fields=FEATURE_FIELDS,
            missing_strategy="impute",
        )
        features = source_data[FEATURE_FIELDS].astype(float)
        target = source_data["target"].astype(int)
        ablation_results = run_naive_bayes_feature_ablation(
            features,
            target,
            cv_folds=CV_FOLDS,
            random_state=args.random_state,
        )
        ablation_path = args.output_dir / "naive_bayes_feature_ablation.csv"
        ablation_results.to_csv(ablation_path, index=False)
    except Exception as e:
        print(f"Error running Naive Bayes feature ablation: {e}")
    
    print(f"\nDataset: {args.data_path}")
    print(f"Records: {record_count}")
    print(f"Features: {len(FEATURE_FIELDS)}")
    print(f"Validation: {CV_FOLDS}-fold stratified cross-validation\n")
    print(f"Saved comparison data and plots: {args.output_dir}")
    if graph_report is not None:
        print(f"Saved plots and local maxima: {args.output_dir}")
        for metric, maxima in graph_report["local_maxima"].items():
            maxima_text = ", ".join(
                f"K={point['K']} ({point['value']:.2%})" for point in maxima
            ) or "none"
            print(f"Local maxima for {metric}: {maxima_text}")
        print()
    if ablation_results is not None:
        print("Naive Bayes leave-one-feature-out impact (ranked by Delta F1):")
        print(
            ablation_results[
                [
                    "Removed Feature",
                    "Delta Accuracy",
                    "Delta Precision",
                    "Delta Recall",
                    "Delta F1",
                ]
            ].to_string(
                index=False,
                formatters={
                    "Delta Accuracy": "{:.2%}".format,
                    "Delta Precision": "{:.2%}".format,
                    "Delta Recall": "{:.2%}".format,
                    "Delta F1": "{:.2%}".format,
                },
            )
        )
        print()
    print(
        results_for_print.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.2%}".format,
                "Accuracy Std": "{:.2%}".format,
                "Precision": "{:.2%}".format,
                "Recall": "{:.2%}".format,
                "F1": "{:.2%}".format,
                "F1 Std": "{:.2%}".format,
            },
        )
    )
    



if __name__ == "__main__":
    main()
