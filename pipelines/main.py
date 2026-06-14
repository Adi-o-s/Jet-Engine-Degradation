import os
import sys

# Resolve project root so paths work regardless of where the script is run from
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import glob
import yaml
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from src.data.data_loading import load_train_data
from src.transform.data_transformation import RULAdder, ConstantColumnDropper, SequenceCreator, DataScaler

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Argument Parsing
parser = argparse.ArgumentParser(description='Train ML model with parameters from config file.')
parser.add_argument('--config', default='config.yaml', help='Path to config.yaml')
parser.add_argument('--learning_rate', type=float, help='Override learning rate')
parser.add_argument('--epochs', type=int, help='Override number of epochs')
parser.add_argument('--batch_size', type=int, help='Override batch size')
parser.add_argument('--model_path', help='Path to a trained .keras model to evaluate')
args = parser.parse_args()

# Load config file
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# Override parameters if provided in command-line
sequence_length = config['data']['sequence_length']
test_size = config['data']['test_size']
random_state = config['data']['random_state']
learning_rate = args.learning_rate or config['training']['learning_rate']
epochs = args.epochs or config['training']['epochs']
batch_size = args.batch_size or config['training']['batch_size']

def evaluate_model_mae(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f'Test MAE: {mae:.4f}')
    return mae

def resolve_model_path(model_path=None):
    """
    Determine which trained model to evaluate.

    Resolution order:
    1. An explicit --model_path passed on the command line.
    2. The most recently saved model under ./models/saved.
    3. The bundled model.keras at the project root.
    """
    if model_path:
        return model_path

    saved_models = glob.glob(os.path.join(PROJECT_ROOT, 'models', 'saved', '*.keras'))
    if saved_models:
        return max(saved_models, key=os.path.getmtime)

    return os.path.join(PROJECT_ROOT, 'model.keras')

if __name__ == "__main__":

    # Data Transformation
    rul_adder = RULAdder()
    raw_data_with_rul = rul_adder.transform(load_train_data())

    constant_column_dropper = ConstantColumnDropper()
    transformed_data = constant_column_dropper.fit_transform(raw_data_with_rul)

    sequence_creator = SequenceCreator(sequence_length=sequence_length)
    sequences, labels = sequence_creator.transform_with_labels(transformed_data)

    # Data Splitting and Scaling
    X_train, X_test, y_train, y_test = train_test_split(sequences, labels, test_size=test_size, random_state=random_state)

    data_scaler = DataScaler()
    X_train_scaled = data_scaler.fit_transform(X_train)
    X_test_scaled = data_scaler.transform(X_test)

    # Load the trained model
    model_path = resolve_model_path(args.model_path)
    print(f'Loading model from: {model_path}')
    model = tf.keras.models.load_model(model_path)

    # Evaluate the model for MAE
    mae = evaluate_model_mae(model, X_test_scaled, y_test)
    print(f'The final MAE is: {mae}')
