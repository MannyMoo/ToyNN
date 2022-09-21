import numpy as np


def sigmoid(z):
    '''Calculate the sigmoid function on the input array, z.'''
    return 1./(1+np.exp(-z))


def linear(x):
    '''Linear activation function: just returns x.'''
    return x


# ReLU activation function
ReLU = np.vectorize(lambda x : max(x, 0))


def softmax(z):
    '''Softmax activation function.'''
    m = z.shape[0]
    ez = np.exp(z)
    ezsum = np.sum(ez, axis=1).reshape((m, 1))
    return ez/ezsum


def mse(ypred, y):
    '''Calculate mean squared error for a given set of predictions and labels.'''
    deltas = ypred - y
    m = ypred.shape[0]
    return np.dot(deltas.T, deltas)/2/m


def log_loss(ypred, y):
    '''Calculate the log loss (binary cross entropy) of the given set of
    predictions and labels.'''
    m = ypred.shape[0]
    return (np.dot(y.T, -np.log(ypred)) + np.dot((1-y).T, -np.log(1-ypred)))/m


def softmax_loss(ypred, y):
    '''Calculate the softmax (sparse categorical cross entropy) of
    the given predictions and labels (with integer values).'''
    m = ypred.shape[0]
    ypred = np.array([ypr[i] for ypr, i in zip(ypred, y)])
    val = np.sum(-np.log(ypred))/m
    return [[val]]
        

class Layer(object):
    '''A simple neural network layer.'''
    
    def __init__(self, nfeatures, nneurons, activation):
        '''Takes number of features (nfeatures), number of neurons (nneurons),
        and a vectorized activation function (activation).'''
        # initialise weights randomly between -1 and 1
        # Each column of self.weights corresponds to the weights for each neuron
        self.weights = np.random.random((nfeatures, nneurons))*2 - 1.
        # initialise biases to zero
        self.biases = np.zeros((1, nneurons))
        self.activation = activation

    def __call__(self, features):
        '''Evaluate the output of the layer.'''
        # (m x nfeat) (nfeat x nneur) + (1 x nneur)
        # = (m x nneur) + (1 x nneur)
        # Uses broadcasting to repeat self.biases into m rows
        return self.activation(np.dot(features, self.weights) + self.biases)


class Network(object):
    '''A simple neural network.'''
    
    def __init__(self, shapes, activations, seed=0):
        '''Takes the list of shapes of each layer (number of features in the input layer,
        number of neurons in the following hidden layers, number of outputs in the final layer),
        and the list of activation functions for each hidden layer and the output layer.'''
        np.random.seed(seed)
        self.layers = []
        if not isinstance(activations, (tuple, list)):
            activations = [activations] * (len(shapes)-1)
        for nfeatures, nneurons, activation in zip(shapes[:-1], shapes[1:],
                                                   activations):
            layer = Layer(nfeatures, nneurons, activation)
            self.layers.append(layer)
        self.layers = tuple(self.layers)

    def __call__(self, features):
        '''Evaluate the network on the given features.'''
        for layer in self.layers:
            features = layer(features)
        return features


class Model(object):
    '''A NN model for training & predicting.'''

    def __init__(self, network, loss=mse, learning_rate=0.001,
                 deltagrad=0.001):
        '''Takes the NN.'''
        self.network = network
        self.loss = loss
        self.lr = learning_rate
        self.deltagrad = deltagrad
        
    def __call__(self, features):
        '''Evaluate the network for the given features.'''
        return self.network(features)

    def cost(self, x, y):
        '''Calculate the current value of the cost function.'''
        ypred = self(x)
        return self.loss(ypred, y)[0][0]
    
    def gradients(self, x, y):
        '''Calculate the gradients of the cost function wrt all
        weights and biases.'''
        grads = []
        basecost = self.cost(x, y)
        for layer in self.network.layers:
            for weight in layer.weights, layer.biases:
                for i in range(weight.shape[0]):
                    for j in range(weight.shape[1]):
                        weight[i][j] += self.deltagrad
                        pcost = self.cost(x, y)
                        weight[i][j] -= 2*self.deltagrad
                        mcost = self.cost(x, y)
                        weight[i][j] += self.deltagrad
                        grads.append(((pcost - mcost)/2./self.deltagrad))
        return grads

    def step(self, x, y):
        '''Take a step in gradient descent.'''
        grads = self.gradients(x, y)
        ig = 0
        for layer in self.network.layers:
            for weight in layer.weights, layer.biases:
                for i in range(weight.shape[0]):
                    for j in range(weight.shape[1]):
                        weight[i][j] -= self.lr * grads[ig]
                        ig += 1

    def fit(self, x, y, epochs, printfreq=1):
        '''Fit to labelled data.'''
        costs = []
        for i in range(epochs):
            self.step(x, y)
            cost = self.cost(x, y)
            costs.append(cost)
            if i%printfreq==0:
                print(f'Epoch {i}/{epochs}, cost = {cost:.3f}')
        return costs

    
def fit(model, x, y, epochs):
    '''Fit a simple model to labelled data using the given number of epochs.'''
    print(f'w = {model.network.layers[0].weights[0][0]:.2f}, b = {model.network.layers[0].biases[0][0]:.2f}')
    print(f'predictions: {model(x)}')
    print('Cost:', model.cost(x, y))
    costs = model.fit(x, y)
    print(f'w = {model.network.layers[0].weights[0][0]:.2f}, b = {model.network.layers[0].biases[0][0]:.2f}')
    print(f'predictions: {model(x)}')
    
                        
def test_linear():
    '''Test simple linear regression.'''
    nn = Network([1, 1], linear)
    model = Model(nn, learning_rate=0.006)
    m = 10
    # Expect w = 1, b = 0
    x = np.array(range(m)).reshape((m, 1))
    y = np.array(range(m)).reshape((m, 1))
    
    epochs = 100
    fit(model, x, y, epochs)
    return model, x, y


def test_classification():
    '''Test simple logistic classification.'''
    nn = Network([1, 1], sigmoid)
    model = Model(nn, loss=log_loss, learning_rate=1.)

    m = 10
    # Expect w = large, b/w = -5
    x = np.array(range(m)).reshape((m, 1))
    y = np.array([0] * int(m/2) + [1] * int(m/2)).reshape((m, 1))

    epochs = 100
    fit(model, x, y, epochs)
    return model, x, y


def test_classification_2D():
    nn = Network([2, 3, 1],
                 [ReLU, sigmoid]
                 # sigmoid
                 )
    model = Model(nn, loss=log_loss, learning_rate=1)

    m = 100
    x = np.random.random((m, 2))

    def test(example):
        a1, a2 = example
        return a1 < 0.1 or a2 < 0.1 or a1 + a2 > 1.

    y = np.array([int(test(ex)) for ex in x]).reshape((m, 1))

    epochs = 600
    model.fit(x, y, epochs, 50)
    ypred = model(x)
    print()
    print('Label, prediction:')
    for _y, _ypred in zip(y, ypred):
        print(_y[0], _ypred[0])
    print()
        
    layer = model.network.layers[0]
    weights = layer.weights.T
    for i in range(weights.shape[0]):
        print('neuron', i, 'weights', weights[i], 'bias', layer.biases[0][i])
    print()

    print('Prediction map:')
    xvals = [0.1*i for i in range(11)]
    yvals = xvals[::-1]
    for yval in yvals:
        print(f'{yval:3.1f} |', end=' ')
        for xval in xvals:
            p = model(np.array([xval, yval]).reshape(1, 2))[0][0]
            print(f'{p:4.2f}', end=' ')
        print()
    print('-'*(6 + 5*11))
    print('y/x |', end=' ')
    for xval in xvals:
        print(f'{xval:4.1f}', end=' ')
    print()
        
    return model, x, y


def test_multiclass():
    '''Test a multi-class classification problem.'''
    nn = Network([2, 3, 4],
                 [ReLU, softmax]
                 )
    model = Model(nn, loss=softmax_loss, learning_rate=1)

    m = 100
    x = np.random.random((m, 2))

    def test(example):
        a1, a2 = example
        if a1 < 0.5:
            if a2 < 0.5:
                return 0
            return 1
        if a2 > 0.5:
            return 2
        return 3

    y = np.array([int(test(ex)) for ex in x]).reshape((m, 1))

    epochs = 600
    model.fit(x, y, epochs, 50)
    ypred = model(x)
    print()
    print('Label, prediction:')
    for _y, _ypred in zip(y, ypred):
        print(_y[0], _ypred)
    print()
        
    layer = model.network.layers[0]
    weights = layer.weights.T
    for i in range(weights.shape[0]):
        print('neuron', i, 'weights', weights[i], 'bias', layer.biases[0][i])
    print()

    print('Prediction map:')
    xvals = [0.1*i for i in range(11)]
    yvals = xvals[::-1]
    for yval in yvals:
        print(f'{yval:3.1f} |', end=' ')
        for xval in xvals:
            p = model(np.array([xval, yval]).reshape(1, 2))[0]
            p = max(enumerate(p), key=lambda x : x[1])[0]
            print(f'{p:>4}', end=' ')
        print()
    print('-'*(6 + 5*11))
    print('y/x |', end=' ')
    for xval in xvals:
        print(f'{xval:4.1f}', end=' ')
    print()
        
    return model, x, y
    

if __name__ == '__main__':
    # nn = Network([3,8,4,1], sigmoid)
    # model = Model(nn, log_loss)
    # data = np.array([[1,2,3],
    #                     [3,4,5]])
    # labels = np.array([0, 1])
    # print(model(data))
    # print(model.cost(data, labels))

    # test_linear()

    # model, x, y = test_classification()

    # model, x, y = test_classification_2D()

    model, x, y = test_multiclass()
