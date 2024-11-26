import numpy as np

class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Initializes a linear (fully connected) layer.
        """
        self.params = {
            'weight': np.random.randn(in_features, out_features) * 0.01,  # Small random values
            'bias': np.zeros((1, out_features))  # Zeros
        }
        self.grads = {
            'weight': np.zeros_like(self.params['weight']),
            'bias': np.zeros_like(self.params['bias'])
        }

    def forward(self, x):
        """
        Performs the forward pass using the formula: output = xW + b
        """
        self.x = x  # Store input for backward pass
        return np.dot(x, self.params['weight']) + self.params['bias']

    def backward(self, dout):
        """
        Backward pass to calculate gradients of loss w.r.t. weights and inputs.
        """
        self.grads['weight'] = np.dot(self.x.T, dout)
        self.grads['bias'] = np.sum(dout, axis=0, keepdims=True)
        return np.dot(dout, self.params['weight'].T)

class ReLU(object):
    def forward(self, x):
        """
        Applies the ReLU activation function element-wise to the input.
        """
        self.mask = (x > 0)
        return np.maximum(0, x)

    def backward(self, dout):
        """
        Computes the gradient of the ReLU function.
        """
        return dout * self.mask

class SoftMax(object):
    def forward(self, x):
        """
        Applies the softmax function to the input to obtain output probabilities.
        """
        shift_x = x - np.max(x, axis=1, keepdims=True)  # Max trick for numerical stability
        exp_x = np.exp(shift_x)
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output

    def backward(self, dout):
        """
        The backward pass for softmax is often directly integrated with CrossEntropy for simplicity.
        """
        return dout

class CrossEntropy(object):
    def forward(self, x, y):
        """
        Computes the CrossEntropy loss between predictions and true labels.
        """
        self.p = x
        self.y = y
        return -np.sum(y * np.log(self.p + 1e-13)) / y.shape[0]  # 防置log(0)出现，13表现比较好

    def backward(self, x, y):
        """
        Computes the gradient of CrossEntropy loss with respect to the input.
        """
        return self.p - y  # 这次不用self.y_pred - self.y_true了
