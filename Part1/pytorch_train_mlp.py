from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import torch
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons
from pytorch_mlp import MLP

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

FLAGS = None

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the average of correct predictions
    of the network.
    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
    Returns:
        accuracy: scalar float, the accuracy of predictions.
    """
    correct = (predictions == targets).float()
    accuracy = correct.mean().item()
    return accuracy


def train(dnn_hidden_units, learning_rate, max_steps, eval_freq):
    """
    Performs training and evaluation of MLP model.
    NOTE: Evaluates the model on the whole test set every eval_freq iterations.
    Returns:
        train_accuracies: List of training accuracies over the epochs.
        test_accuracies: List of test accuracies over the epochs.
    """
    if isinstance(dnn_hidden_units, str):
        dnn_hidden_units = [int(unit) for unit in dnn_hidden_units.split(',')]

    model = MLP(n_inputs=2, n_hidden=dnn_hidden_units, n_classes=2)  # n_inputs is 2 for make_moons data
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    # Generate data
    train_loader, x_test, y_test = generate_data()

    train_accuracies = []
    test_accuracies = []

    for epoch in range(max_steps):
        model.train()
        correct_train = 0
        total_train = 0
        for x_batch, y_batch in train_loader:
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            predictions = torch.argmax(outputs, dim=1)
            correct_train += (predictions == y_batch).sum().item()
            total_train += y_batch.size(0)

        # Calculate and store training accuracy
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        if epoch % eval_freq == 0:
            model.eval()
            with torch.no_grad():
                outputs = model(x_test)
                predictions = torch.argmax(outputs, dim=1)
                test_accuracy = accuracy(predictions, y_test)
                test_accuracies.append(test_accuracy)

            print(
                f"Epoch {epoch}, Loss: {loss.item()}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

    return train_accuracies, test_accuracies

def generate_data():
    """
    Generates synthetic training and test data using make_moons.
    Returns:
        train_loader: DataLoader for the training set.
        x_test: Tensor of test data features.
        y_test: Tensor of test data labels.
    """
    # Generate data using make_moons
    X, y = make_moons(n_samples=1000, noise=0.1)

    # Split into training and testing sets
    x_train = torch.tensor(X[:800], dtype=torch.float32)  # 800 samples for training
    y_train = torch.tensor(y[:800], dtype=torch.long)
    x_test = torch.tensor(X[800:], dtype=torch.float32)   # 200 samples for testing
    y_test = torch.tensor(y[800:], dtype=torch.long)

    # Create DataLoader for batching
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    return train_loader, x_test, y_test

def main():
    """
    Main function
    """
    train()

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()
