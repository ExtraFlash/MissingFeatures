import torch
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from dae import DAEModel
import ActivationFactory


# Define the DAELightning class (replace with the exact architecture you used during training)

# Function to preprocess data
# def preprocess_data(df):
#     """
#     Preprocess the data by replacing missing values with 0.
#     You can modify this to use other imputation strategies (e.g., column mean).
#     """
#     preprocessed_df = df.fillna(0)  # Replace NaN values with 0
#     return preprocessed_df

# Function to complete missing data using the trained model
def complete_missing_data(model_path, data_file, output_file, input_size, hidden_size):
    """
    Completes missing data using the trained model.

    Args:
        model_path: Path to the saved model file (e.g., "dae_model_entire.pth").
        data_file: Path to the CSV file containing the input data with missing values.
        output_file: Path to save the completed data.
        input_size: Number of input features (columns in the dataset).
        hidden_size: Size of the hidden layer in the DAE model.
    """
    # Load the model
    model = DAEModel(input_size=input_size,
                     latent_dim=35,
                     encoder_units=(146, 103),  # encoder -> latent_dim -> decoder (1 hidden layer each)
                     decoder_units=(129, 170),
                     dropout_rate=0.436,
                     learning_rate=1.961572097160407e-05,
                     activation_name=ActivationFactory.relu_NAME,
                     should_support_missing_values=True,
                     )

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()  # Set the model to evaluation mode

    # Load the dataset
    df = pd.read_csv(data_file)

    # Preprocess the dataset
    preprocessed_df = preprocess_data(df)

    # Convert preprocessed data to a PyTorch tensor
    input_tensor = torch.tensor(preprocessed_df.values, dtype=torch.float32)

    # Perform imputation using the model
    with torch.no_grad():
        completed_data = model(input_tensor).numpy()

    # Create a DataFrame with the completed data
    completed_df = pd.DataFrame(completed_data, columns=df.columns)

    # Save the completed data to a CSV file
    completed_df.to_csv(output_file, index=False)
    print(f"Completed data saved to {output_file}")


# Example usage
if __name__ == "__main__":
    # Path to the trained model file
    model_path = "dae_model_entire.pth"

    # Path to the input data file (with missing values)
    data_file = "IUGR_All_OH_No_MOM.csv"

    # Path to save the completed data
    output_file = "completed_data_IUGR.csv"

    # Define the model parameters (adjust these based on your model)
    input_size = 10  # Replace with the actual number of input features
    hidden_size = 5  # Replace with the actual hidden layer size

    # Run the completion process
    complete_missing_data(model_path, data_file, output_file, input_size, hidden_size)
