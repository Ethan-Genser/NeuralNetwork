import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data(path="mnist.npz")

    network1 = Network(784, 10, Softmax(), CategoricalCrossEntropy(), (400,ReLU()), (100,ReLU()))
    network1.forward(np.ndarray.flatten(x_train[0]))
    loss = network1.calculateLoss(5)

    print (f"yPred: {network1.yPrediction} loss: {loss}")

class Network:
    def __init__(self, xDimensions, yDimensions, yActivation, lossFunction, *hiddenLayers):
        self.layers = []
        self.activations = []
        self.lossFunction = lossFunction
        lastLayer = xDimensions
        for (layer, activation) in hiddenLayers:
            self.layers.append(LayerDense(lastLayer, layer))
            self.activations.append(activation)
            lastLayer = layer
        self.layers.append(LayerDense(lastLayer, yDimensions))
        self.activations.append(yActivation)
    def forward(self, x):
        out = x
        for layer, activation in zip(self.layers, self.activations):
            layer.forward(out)
            activation.forward(layer.output)
            out = activation.output
        self.yPrediction = out
    def calculateLoss(self, yAnswer):
        self.lossFunction.forward(self.yPrediction, yAnswer)
        return (self.lossFunction.output)

class LayerDense:
    def __init__(self, inputCount, outputCount, wieghts=[], biases=[]):
        if (not wieghts):
            # Assign weights randomly between -1 and 1 if none are provided
            self.wieghts = np.random.randn(inputCount, outputCount) * 0.1
        elif (np.shape(wieghts) == (inputCount, outputCount)):
            self.wieghts = wieghts
        else:
            raise Exception("invalid shape error: LayerDense.wieghts")
        
        if (not biases):
            # Assign all biases to zero if none are provided
            self.biases = np.zeros((1,outputCount))
        elif (np.shape(biases) == (1, outputCount)):
            self.biases = biases
        else:
            raise Exception("invalid shape error: LayerDense.biases")

    def forward(self, inputs):
        self.output = np.dot(inputs, self.wieghts) + self.biases

class Activation:
    def forward(self, inputs):
        self.output = inputs

class ReLU(Activation):
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Softmax(Activation):
    def forward(self, inputs):
        exp = np.exp(inputs - np.max(inputs))
        self.output = exp / np.sum(exp)

class Loss:
    def calculate(self, yPrediction, yAnswer):
        self.output = self.forward(yPrediction, yAnswer)

class CategoricalCrossEntropy(Loss):
    def forward(self, yPrediction, yAnswer):
        # Clips prediction values by very small number to avoid log(0) error
        yPredictionClipped = np.clip(yPrediction, 1e-7, 1-1e-7)
        self.output = -np.log(yPrediction[0, yAnswer])
    

if(__name__=="__main__"):main()