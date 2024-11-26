from modules import *

class MLP(object):
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes the multi-layer perceptron object.
        """
        self.layers = []
        # hidden layers
        in_features = n_inputs
        for hidden_units in n_hidden:
            self.layers.append(Linear(in_features, hidden_units))
            self.layers.append(ReLU())
            in_features = hidden_units

        # Create output layer
        self.layers.append(Linear(in_features, n_classes))
        self.layers.append(SoftMax())

    def forward(self, x):
        """
        Predicts the network output from the input by passing it through several layers.
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dout):
        """
        Performs the backward propagation pass given the loss gradients.
        """
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
