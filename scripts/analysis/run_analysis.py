import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from dataset_analysis import Dataset
from config import paths as project_paths


# Open the Cleveland heart disease dataset and load it into a Dataset object
cd = Dataset(
    data_path=project_paths.BUNDLED_DATASETS_DIR / 'heart_disease_cleveland_cleaned.csv',
    short_name='cleveland',
    long_name='Cleveland Heart Disease Dataset'
)


# Load the dataset
cd.load_data()

# Preprocess the dataset
cd.preprocess_data()

# Generate histograms for each column in the dataset
cd.visualize_columns()

# Visualize correlations between features in the dataset
cd.visualize_correlations()
