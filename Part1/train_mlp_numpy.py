import argparse
import numpy as np
from sklearn.datasets import make_moons

from mlp_numpy import MLP
from modules import CrossEntropy, Linear

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-4
MAX_EPOCHS_DEFAULT = 600 # adjust if you use batch or not
EVAL_FREQ_DEFAULT = 10

def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e., the percentage of correct predictions.

    Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        targets: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding

    Returns:
        accuracy: scalar float, the accuracy of predictions as a percentage.
    """
    # TODO: Implement the accuracy calculation
    # Hint: Use np.argmax to find predicted classes, and compare with the true classes in targets
    preds = np.argmax(predictions, axis=1)
    true = np.argmax(targets, axis=1)
    return np.mean(preds == true) * 100  # Return percentage
def train(dnn_hidden_units, learning_rate, max_steps, eval_freq, use_stochastic=False, batch_size=32):
    """
    Performs training and evaluation of MLP model.

    Args:
        dnn_hidden_units: Comma separated list of number of units in each hidden layer
        learning_rate: Learning rate for optimization
        max_steps: Number of epochs to run trainer
        eval_freq: Frequency of evaluation on the test set
        NOTE: Add necessary arguments such as the data, your model...
    """
    X, y = make_moons(n_samples=1000, noise=0.1)
    y_onehot = np.eye(2)[y]  # One-hot encoding for binary classification

    # Split the data into training (80%) and testing (20%) sets
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    train_size = int(0.8 * n_samples)
    train_indices, test_indices = indices[:train_size], indices[train_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y_onehot[train_indices], y_onehot[test_indices]

    # Initialize MLP model and loss function (CrossEntropy)
    dnn_hidden_units = list(map(int, dnn_hidden_units.split(',')))
    mlp = MLP(n_inputs=X_train.shape[1], n_hidden=dnn_hidden_units, n_classes=y_train.shape[1])
    loss_fn = CrossEntropy()

    # for notebook
    train_accuracies = []
    test_accuracies = []

    for step in range(max_steps):
        if not use_stochastic:
            # Training loop
            indices = np.arange(train_size)
            np.random.shuffle(indices)  # Shuffle data for each epoch

            for start in range(0, train_size, batch_size):  # Example batch size of 32
                end = min(start + batch_size, train_size)
                batch_indices = indices[start:end]

                # Forward pass
                predictions = mlp.forward(X_train[batch_indices])
                loss = loss_fn.forward(predictions, y_train[batch_indices])

                # Backward pass (compute gradients)
                dout = loss_fn.backward(predictions, y_train[batch_indices])
                mlp.backward(dout)

                # Update weights
                for layer in mlp.layers:
                    if isinstance(layer, Linear):
                        layer.params['weight'] -= learning_rate * layer.grads['weight']
                        layer.params['bias'] -= learning_rate * layer.grads['bias']
        else: # stohastic
            # Stochastic Gradient Descent
            for i in range(train_size):
                # Get the index for the current sample
                batch_indices = [i]

                # Forward pass
                predictions = mlp.forward(X_train[batch_indices])
                loss = loss_fn.forward(predictions, y_train[batch_indices])

                # Backward pass (compute gradients)
                dout = loss_fn.backward(predictions, y_train[batch_indices])
                mlp.backward(dout)

                # Update weights
                for layer in mlp.layers:
                    if isinstance(layer, Linear):
                        layer.params['weight'] -= learning_rate * layer.grads['weight']
                        layer.params['bias'] -= learning_rate * layer.grads['bias']
        # 记录训练准确率
        train_preds = mlp.forward(X_train)
        train_acc = accuracy(train_preds, y_train)
        train_accuracies.append(train_acc)
        # Evaluate the model
        if step % eval_freq == 0 or step == max_steps - 1:
            val_predictions = mlp.forward(X_test)
            val_loss = loss_fn.forward(val_predictions, y_test)
            val_acc = accuracy(val_predictions, y_test)
            test_accuracies.append(val_acc) # for notebook
            print(f"Step: {step}, Loss: {loss}, Test Loss: {val_loss}, Test Accuracy: {val_acc}%")

    print("Training complete!")
    print(train_accuracies)
    print(test_accuracies)
    return [train_accuracies, test_accuracies]

def main():
    """
    Main function.
    """
    # Parsing command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_EPOCHS_DEFAULT,
                        help='Number of epochs to run trainer')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    FLAGS = parser.parse_known_args()[0]

    train(FLAGS.dnn_hidden_units, FLAGS.learning_rate, FLAGS.max_steps, FLAGS.eval_freq)

if __name__ == '__main__':
    main()
