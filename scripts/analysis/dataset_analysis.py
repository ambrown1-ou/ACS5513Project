from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOTS_DIR = PROJECT_ROOT / 'static' / 'plots'
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class Dataset:
    '''Class to handle a dataset by loading, preprocessing, and providing features and target variables. 
    '''
    def __init__(self, data_path: str, short_name: str, long_name: str):
        self.short_name = short_name
        self.data_path = data_path
        self.long_name = long_name
        self.df = None
        self.output_dir = PLOTS_DIR
        
        # Standard fields from Cleveland heart disease dataset
        self.standardized_fields = [
            'age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'target'
        ]

        self.field_definitions = {
            'age': 'Age in years',
            'trestbps': 'Resting blood pressure (in mm Hg on admission to the hospital)',
            'chol': 'Serum cholesterol in mg/dl',
            'thalach': 'Maximum heart rate achieved',
            'oldpeak': 'ST depression induced by exercise relative to rest',
            'sex': 'Sex (1 = male; 0 = female)',
            'cp': 'Chest pain type',
            'fbs': 'Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
            'restecg': 'Resting electrocardiographic results',
            'exang': 'Exercise induced angina (1 = yes; 0 = no)',
            'slope': 'Slope of the peak exercise ST segment',
            'ca': 'Number of major vessels (0-3) colored by fluoroscopy',
            'thal': 'Thalassemia',
            'target': 'Presence of heart disease (1 = yes; 0 = no)'
        }

    def get_standardized_fields(self):
        """Return the standardized fields for the dataset."""
        return self.standardized_fields
    
    def load_data(self):
        """Load the Cleveland heart disease dataset."""
        try:
            self.df = pd.read_csv(self.data_path)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def preprocess_data(self, field_mapping: dict = None, drop_fields: list = None):
        """Preprocess the dataset by handling missing values and encoding categorical variables."""
        if self.df is not None:
            # Handle missing values
            self.df.replace('?', np.nan, inplace=True)
            self.df.dropna(inplace=True)

            # Convert categorical variables to numeric
            categorical_columns = self.df.select_dtypes(include=['object']).columns

            # Convert categorical variables to numeric using category codes
            # lambda x: x.astype('category') is used to convert the column to a categorical type
            self.df[categorical_columns] = self.df[categorical_columns].apply(lambda x: x.astype('category').cat.codes) 

            # Get the standardized fields that are present in the dataset

            # First check field mapping if provided and rename columns accordingly
            if field_mapping is not None:
                self.df.rename(columns=field_mapping, inplace=True)
            self.standardized_fields = [field for field in self.standardized_fields if field in self.df.columns]

            if len(self.standardized_fields) < 14:
                print(f"Warning: Some standardized fields are missing in the dataset. Available fields: {self.standardized_fields}")

            # Drop specified fields if any
            if drop_fields is not None:
                self.df.drop(columns=drop_fields, inplace=True, errors='ignore')

    def visualize_columns(self):
        """Generate histograms for each column in the dataset."""
        if self.df is not None:
            self.df.hist(figsize=(12, 10))
            
            #make sure to add a title for each histogram and a main title for the entire figure
            for ax in plt.gcf().axes:
                ax.set_title(ax.get_title().replace('_', ' ').title())
            plt.suptitle(f'Histograms for {self.long_name}')
            
            # Export to output file.
            output_path = self.output_dir / f'{self.short_name}_histograms.png'
            plt.savefig(output_path)
            plt.close()
            print(f"Histograms saved to {output_path}")

        
    def visualize_correlations(self):
        """Find and generate a chart of the correlations between features in the dataset."""
        if self.df is not None:
            plt.figure(figsize=(12, 10))
            correlation_matrix = self.df.corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title(f'Correlation Matrix for {self.long_name}')
            
            # Export to output file.
            output_path = self.output_dir / f'{self.short_name}_correlation_matrix.png'
            plt.savefig(output_path)
            plt.close()
            print(f"Correlation matrix saved to {output_path}")
            plt.figure(figsize=(12, 10))
            plt.title(f'Correlation Matrix for {self.long_name}')
            plt.savefig(PROJECT_ROOT / 'outputs' / f'{self.short_name}_correlation_matrix.png')
            plt.show()
