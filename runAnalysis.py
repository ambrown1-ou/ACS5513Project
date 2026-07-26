from processInputs import Dataset

# Open the Cleveland heart disease dataset and load it into a Dataset object
cd = Dataset(
    data_path='inputs/Heart_disease_cleveland_new.csv', 
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

