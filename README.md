# OU ACS-5513: Heart Disease Clinical Dashboard

This project is a clinical diagnostic dashboard developed for the University of Oklahoma ACS-5513 Machine Learning course. It provides an end-to-end pipeline for exploring heart disease data, training predictive models (KNN, Naive Bayes, SVM), and deploying them for individual diagnostic testing.

## Project Scope
- **Interactive EDA**: Visualizing clinical feature distributions and correlations using histograms and Plotly-based 3D scatter plots.
- **Clinical Methodology**: A modular training interface allowing users to benchmark different classification algorithms.
- **Model Deployment**: A functional interface for clinical practitioners to input patient metrics and receive diagnostic probabilities.

## Local Setup & Execution

### Prerequisites
- Python 3.12+
- Virtual environment (recommended)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/ACS5513Project/acs5513.git
   cd acs5513
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
To start the Flask web server:
```bash
python run.py
```
The application will be available at `http://127.0.0.1:5000`.

## Data Dictionary

| Field | Domain / Scale | Units | Description |
| :--- | :--- | :--- | :--- |
| `age` | Numeric (29-77) | years | Age of the patient. |
| `sex` | 0=Female, 1=Male | binary | Biological sex of the patient. |
| `cp` | 1=Typ, 2=Atyp, 3=Non, 4=Asymp | coded | Chest pain type experienced. |
| `trestbps` | Numeric (94-200) | mm Hg | Resting blood pressure on admission. |
| `chol` | Numeric (126-564) | mg/dl | Serum cholesterol measurements. |
| `fbs` | >120 mg/dl (1=True, 0=False) | binary | Fasting blood sugar level. |
| `restecg` | 0=Norm, 1=ST-T, 2=LVH | coded | Resting electrocardiographic results. |
| `thalach` | Numeric (71-202) | bpm | Maximum heart rate achieved. |
| `exang` | 1=Yes, 0=No | binary | Exercise induced angina. |
| `oldpeak` | Numeric (0-6.2) | mm | ST depression induced by exercise relative to rest. |
| `slope` | 1=Up, 2=Flat, 3=Down | coded | Slope of the peak exercise ST segment. |
| `ca` | 0-3 vessels | count | Number of major vessels colored by fluoroscopy. |
| `thal` | 3=Norm, 6=Fixed, 7=Rev | coded | Thalassemia (blood disorder) status. |
| `target` | 0=Healthy, 1=Disease | binary | Final diagnosis status. |

---
*Created for University of Oklahoma - ACS-5513: Machine Learning*
