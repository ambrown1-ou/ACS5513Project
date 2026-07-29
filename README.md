# OU ACS-5513: Heart Disease Clinical Dashboard

This project is a Flask-based clinical dashboard for exploring heart disease datasets, training supervised classification models, comparing their results, and running patient-level predictions. It is currently built around the cleaned Cleveland heart disease dataset.

## What The Application Does

1. Explore the bundled dataset with distribution and correlation visualizations.
2. Train supervised models on the selected dataset.
3. Compare training results across methods.
4. Use a trained model to make an individual prediction.

## Python Stack

The project uses the scientific Python ecosystem at a high level:

- `pandas` for loading, cleaning, and shaping tabular dataset files.
- `scikit-learn` for model training, validation splitting, and evaluation metrics.
- `matplotlib`, `seaborn`, and `plotly` for static and interactive visualizations.
- `Flask` and `gunicorn` for the web application and Heroku runtime.
- `joblib` for saving and loading trained model files.
- `JupyterLab` for local data exploration and experimentation.
- `statsmodels` for supporting statistical analysis workflows.

## Application Architecture

The application follows a simple environment flow:

1. Local development: the Flask app runs on a developer machine through `python run.py`; pass `--jupyter` to start JupyterLab alongside it for analysis and experimentation.
2. QA/UAT: code is published through GitHub and deployed to a Heroku QA/UAT dyno for functional testing and validation.
3. Production: once testing passes, the same Heroku release is promoted to the production dyno.

Behind that release flow, the code is organized into a web layer (`app/routes.py` and `app/api.py`), an ML layer (`model/pipeline.py`), visualization helpers, templates, static assets, and file-based inputs and outputs. Trained artifacts are registered with a unique model ID in `outputs/training_results.json`; the API uses the most recently registered model.

At present, dataset ingestion, dataset uploads, and model retraining are supported in the local development environment only. The Heroku QA/UAT and production dynos are used for application validation and prediction workflows.

JupyterLab is only launched in the local development flow when `--jupyter` is supplied; Heroku does not open JupyterLab and instead runs the Flask app through `gunicorn app:app` from the `Procfile`.

## Deployment

The Heroku deployment mirrors the environment flow above: QA/UAT is validated first, then the release is promoted to production after approval.

## Repository Structure

The root directory stays focused on app startup, deployment, and documentation. Reusable command-line scripts live under `scripts/`, split by purpose.

| Path | Purpose |
| --- | --- |
| `app/` | Flask app factory, web routes, and API routes |
| `model/` | Training, prediction, and visualization helpers |
| `scripts/training/` | CLI helpers for single-model and batch training |
| `scripts/analysis/` | Exploratory dataset analysis script and helper module |
| `templates/` | Jinja2 pages for the dashboard |
| `static/` | CSS and generated plot assets |
| `inputs/` | Bundled cleaned Cleveland CSV dataset |
| `outputs/` | Saved models and training results |
| `run.py` | Local development entry point |
| `Procfile` | Heroku web process definition |
| `datasets_info.json` | Dataset metadata used by the UI |
| `PROJECT_STRUCTURE.md` | Detailed directory and file responsibility map |

## Local Setup

### Prerequisites

- Python 3.12+
- A virtual environment is recommended

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Locally

```bash
python run.py
```

The app will be available at `http://127.0.0.1:5000`.

To also start JupyterLab during local development, run:

```bash
python run.py --jupyter
```

### Train a Model from the Command Line

```bash
python scripts/training/train_model.py inputs/heart_disease_cleveland_cleaned.csv --method knn
```

### Train All Models

```bash
python scripts/training/train_all.py
```

### Run the Example Analysis Script

```bash
python scripts/analysis/run_analysis.py
```

## Command Line Interfaces

### `run.py`

Launches the Flask application.

- `--jupyter`: starts JupyterLab alongside the Flask app in the project root.

### `scripts/training/train_model.py`

Trains one model on one dataset.

- `data_path`: optional positional argument for the CSV file to train on. The default is `inputs/heart_disease_cleveland_cleaned.csv`.
- `--method`: training algorithm to use. Supported values are `knn`, `naive_bayes`, and `svm`.
- `--cv-folds`: number of stratified cross-validation folds. The default is `5`.
- `--random-state`: random seed used for the train/test split and model initialization. The default is `42`.
- `--neighbors`: number of neighbors used by KNN. The default is `5`.

### `scripts/training/train_all.py`

Trains every supported method on the bundled cleaned Cleveland dataset.

- No command-line arguments.

### `scripts/analysis/run_analysis.py`

Runs the example exploratory analysis script for the bundled cleaned Cleveland dataset.

- No command-line arguments.

## Data Dictionary

| Field | Units | Domain | Description |
| --- | --- | --- | --- |
| `age` | years | Numeric (0-120) | Age of the patient |
| `sex` | binary | 0 = Female, 1 = Male | Biological sex of the patient |
| `cp` | coded | 1 = Typical, 2 = Atypical, 3 = Non-anginal, 4 = Asymptomatic | Chest pain type |
| `trestbps` | mm Hg | Numeric (0-300) | Resting blood pressure on admission |
| `chol` | mg/dl | Numeric (0-1000) | Serum cholesterol |
| `fbs` | binary | >120 mg/dl (1 = True, 0 = False) | Fasting blood sugar |
| `restecg` | coded | 0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy | Resting electrocardiogram results |
| `thalach` | bpm | Numeric (0-250) | Maximum heart rate achieved |
| `exang` | binary | 1 = Yes, 0 = No | Exercise-induced angina |
| `oldpeak` | mm | Numeric (0-10) | ST depression induced by exercise relative to rest |
| `slope` | coded | 1 = Upsloping, 2 = Flat, 3 = Downsloping | Slope of the peak exercise ST segment |
| `ca` | count | 0-3 vessels | Number of major vessels colored by fluoroscopy |
| `thal` | coded | 3 = Normal, 6 = Fixed defect, 7 = Reversible defect | Thalassemia status |
| `target` | binary | 0 = Healthy, 1 = Heart disease | Diagnosis label used for supervised learning |

## Terms Dictionary

| Term | Meaning |
| --- | --- |
| `supervised learning` | Training a model on labeled examples so it can predict unseen cases |
| `target` | The label the model learns to predict; here it is the heart disease diagnosis |
| `KNN` | K-Nearest Neighbors; predicts by majority vote among the nearest training examples |
| `Naive Bayes` | A probabilistic classifier that scores classes using Bayes' theorem with feature independence assumptions |
| `SVM` | Support Vector Machine; learns a separating boundary between the two classes |
| `prediction` | The class returned by the trained model, where `0` means healthy and `1` means heart disease |
| `probability` | The model's confidence for the predicted class |
| `accuracy` | Share of test cases classified correctly |
| `precision` | Share of predicted positive cases that were actually positive |
| `recall` | Share of actual positive cases that were found by the model |
| `f1` | Harmonic mean of precision and recall |
| `rows` | Number of complete rows used after cleaning the dataset |
| `model_id` | Unique registered identifier for a trained model artifact |
| `model_path` | Project-relative path where the trained `.joblib` model is saved |
| `method` | The algorithm used for training, such as `knn`, `naive_bayes`, or `svm` |
| `name` | The saved label for the trained model file |
| `cv_folds` | Number of stratified folds used for evaluation |
| `accuracy_std`, `precision_std`, `recall_std`, `f1_std` | Standard deviation across cross-validation folds |
| `QA/UAT` | Validation environment used before promoting the release to production |
| `Heroku dyno` | The container that runs the deployed application on Heroku |

## Model Output Summary

The prediction screen returns a binary diagnosis plus confidence. Training reports mean metrics and standard deviations across stratified cross-validation folds, then fits the saved estimator on all validated rows. KNN and SVM use feature scaling within their saved pipelines.

- Prediction output: `prediction` and `probability`
- Training output: `model_id`, `dataset_key`, `accuracy`, `accuracy_std`, `precision`, `precision_std`, `recall`, `recall_std`, `f1`, `f1_std`, `rows`, `cv_folds`, `model_path`, `method`, `name`

## Training Methods

The dashboard compares three classification methods. Each method learns from the labeled patient records and predicts whether a new record belongs to the healthy or heart disease class.

### K-Nearest Neighbors (KNN)

KNN predicts a patient by looking for the most similar patients in the training data. It then uses the majority class among the selected neighbors. The `K` value controls how many neighbors are considered; the dashboard uses `5` by default.

KNN is easy to understand because its decisions are based on nearby examples. It can be affected by unusual records and by the choice of `K`, so the dashboard scales the measurements before comparing distances.

### Naive Bayes

Naive Bayes estimates how likely each class is given the patient measurements, then selects the class with the stronger overall probability. It is fast and often works well on smaller datasets.

Its main limitation is that it treats the measurements as if they contribute independently. Clinical measurements can be related to one another, so its results should be compared with the other methods rather than treated as automatically superior.

### Support Vector Machine (SVM)

SVM looks for a decision boundary that separates the two classes while leaving as much space as possible between them. The configuration used by this project can represent curved as well as straight boundaries, which helps when the classes do not separate cleanly using one simple line.

SVM can produce strong results, but its decisions are less intuitive to explain than KNN decisions. Like KNN, it uses scaled measurements so features with larger numeric ranges do not dominate the model.

### Comparing The Methods

The dashboard evaluates each method with stratified cross-validation, which keeps the class balance similar across the evaluation groups. The displayed score is the average across those groups, and the `+/-` value shows how much the score changes between them. Higher scores are useful, but precision, recall, and F1 should be considered together because a model that is strong on only one measure may still miss important cases.

---
Created for the University of Oklahoma ACS-5513 Machine Learning course.
