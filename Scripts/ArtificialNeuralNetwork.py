import pickle
import numpy as np
from sklearn.model_selection import KFold

"""
A normalisation layer that can be added between two layers:
the main reason for this is to reduce internal covariate shift
Small mini batches are used to normalise the data
This can lead to accelerated leraning and can reduce overfitting
"""
class BatchNormalisationLayer:
    def __init__(self, gamma, beta, eps = 1e-10):
        # parameters of normalisation
        self.gamma = gamma
        self.beta = beta
        # small value applied to calculations to add stability
        self.eps = eps


        # FOR RMS PROP
        # derivate of gamma and beta
        # used in rms prop  to find better values of gamma and beta
        self.dGamma = None
        self.dBeta = None
        # Initialize moving average squared gradients to zero
        self.vdGamma = np.ones_like(gamma)
        self.vdBeta = np.ones_like(beta)


        # This is the data used in batch normalisation
        # initally set all to None, before being accesed they will be initialise in foward() 
        self.set_cache(None, None, None, None, None)

    """
    Set values and intermediary values used in batch normalisation
        x_norm = normalised x
        i_std = inverse of standard deviation (1./std)
        std_batch = standard deviation of batch
        batch_variance = variance of batch
        x_mean = mean of x
    """
    def set_cache(self, x_norm, i_std, std_batch, batch_variance, x_mean):
        self.x_norm = x_norm
        self.i_std = i_std
        self.std_batch = std_batch
        self.batch_variance = batch_variance
        self.x_mean = x_mean

    """
    Foward pass of batch normalisation
    This can be used on the activations of a hidden layer or on the Z values
    Used in foward propagation
    """
    def foward(self, x):
        N, D = x.shape

        # calculate mean of batch
        batch_mean = (1./N) * np.sum(x, axis = 0)
        # subtract this from every sample
        x_mean = x - batch_mean

        # calculate variance of batch
        batch_variance = (1./N) * np.sum(x ** 2, axis = 0)
        # calculate standard deviation
        # eps adds stability
        std_batch = np.sqrt(batch_variance + self.eps)

        # invert standard deviation
        i_std = 1./std_batch

        # normalise
        x_norm = x_mean * i_std

        self.set_cache(x_norm, i_std, std_batch, batch_variance, x_mean)

        # return normalised x scaled by parameters
        # rms prop will learn the best values for gamma and beta
        return self.gamma * x_norm + self.beta
    
    """
    The formulas used here are from the chain rule on the foward pass
    This is essentially to undo the batch normalisation to find the original values
    Used in backwards propagation on either the activations or Z values: depending on what you have done in foward propagation
    """
    def backward(self, x):
        # shape of data
        N, D = x.shape

        # calculate derivative with respect to beta
        dbeta = np.sum(x, axis = 0)

        # calculate derivative with respect to gamma
        dgamma = np.sum(x * self.x_norm, axis = 0)
        dX_norm = x * self.gamma

        # backward propagation into x_norm
        di_std = np.sum(dX_norm * self.x_mean, axis = 0)
        dX_mean1 = dX_norm * self.i_std

        # backward propagation into i_std
        dstd = -1. / (self.i_std ** 2) * di_std

        # backward propagation into std
        dvar = 0.5 * 1. / np.sqrt(self.batch_variance + self.eps) * dstd

        # backward propagation into var
        dsq = 1. / N * np.ones((N, D)) * dvar

        # backward propagation into sq
        dX_mean2 = 2 * self.x_mean * dsq

        # backward proagation dX_mean1 + dX_mean2
        dX1 = (dX_mean1 + dX_mean2)
        dmu = -1 * np.sum(dX_mean1 + dX_mean2, axis = 0)

        # backward propagation mean
        dX2 = 1. / N * np.ones((N,D)) * dmu

        dx = dX1 + dX2

        self.dGamma = dgamma
        self.dBeta = dbeta

        return dx




class ArtificialNeuralNetwork:
    def __init__(self, k):
        self.k = k

    """
    Initialise weights and biases for given network configuration
    """
    def initialise_parameters(self, config):
        # Initialise weights and biases to the given configuration
        # These in a way exsist in between layers
        #np.random.seed(42)
        weights = []
        biases = []
        for i in range(len(config) - 1):
            # config[i+1] = next layer, config[i] = current layer
            # np.random.radn will fill the table with shape (config[i+1], config[i]) with values from normal distribution
            # np.sqrt(2 / config[i]) is an optimisation for better initial weights
            weights.append(np.random.randn(config[i+1], config[i]) * np.sqrt(2 / config[i]))
            biases.append(np.random.randn(config[i+1], 1))
        return weights, biases
    
    """
    Computes the output of the newtork with given weights and biases
    """
    def foward_propagation(self, x, weights, biases, normalisation_layer):
        # Array of intermediary layer to calculate activation layer
        Zarr = []
        # Array of activation layers
        Aarr = []

        # work out intial Z value of dot product + bias
        Zarr.append(weights[0].dot(x) + biases[0])
        # use batch normalisation to normalise the network
        aout = normalisation_layer.foward(Zarr[-1].T)
        # using rectifiedLinearUnit (ReLU) as activation function
        Aarr.append(self.rectifiedLinearUnit(aout.T))

        # for each layer
        for i in range(len(weights)):
            # skip first and last layer as calculated outside
            if i == 0 or i == len(weights) - 1:
                continue
            # product of linear transformation
            Zarr.append(weights[i].dot(Aarr[-1]) + biases[i])
            # ReLU
            Aarr.append(self.rectifiedLinearUnit(Zarr[i]))

        Zarr.append(weights[-1].dot(Aarr[-1]) + biases[-1])
        # Last layer uses different activation function
        # this is known as the normalised exponential function or more commonly as the softmax function
        # softmax function converts a vector of real numbers into a probability distribution
        Aarr.append(self.normalisedExpFunc(Zarr[-1]))
        return Zarr, Aarr, normalisation_layer
    
    """
    This is how the network learns
    Start from and output and work backwards
    The formulas used are derived from the chain rule on the foward pass,
    essentially undoing the foward pass to adjust weights and biases
    """
    def back_propagation(self, x, y, Zarr, Aarr, weights, normalisation_layer):
        # set answer from known answer from training data
        # in other words, work out correct answer for each training set (spam or ham)
        y_out = np.zeros((y.shape[0], len(np.unique(y))))
        print(y_out.shape)
        print(y.shape)
        print(np.arange(y.shape[0]).shape)
        y_out[np.arange(y.shape[0]), y] = 1
        print(y_out.shape)
        y_out = y_out.T 

        # initialise arrays
        dZ = []
        dW = []
        dB = []

        # reverse arrays for simplicity
        # we start from the output layer and work backwards
        Aarr = Aarr[::-1]
        Zarr = Zarr[::-1]
        #n, n-1, n-2, ..., 1
        m = x.shape[0]
        
        # deviation of Z of output layer, used to calculate dW and dB
        # this is deviation of the expected output and softmax output
        dZ.append(2*(Aarr[0] - y_out))
        # deviation of Weights in last weight layer
        dW.append((1/m) * dZ[0].dot(Aarr[1].T))
        # deviation of biases in last bias layer
        dB.append((1/m) * np.sum(dZ[0], 1).reshape(-1, 1))
        
        # deviation is from the expected output from the models predicted output
        # deviation is used to update weights and biases to improve the model
        
        # loop over every layer and work out deviation
        for z, i in zip(Zarr, range(len(Zarr))):
            if i == 0 or i == len(Zarr) - 1:
                continue

            # layer i + 1
            dZ.append(weights[-i].T.dot(dZ[-1]) * self.derivative_ReLU(z))
            dW.append((1/m) * dZ[-1].dot(Aarr[i+1].T))  
            dB.append((1/m) * dZ[-1].sum(axis = 1).reshape(-1, 1))
        
        # deviation of first layer
        dZ.append(weights[1].T.dot(dZ[-1]) * self.derivative_ReLU(Zarr[-1]))
        dH = normalisation_layer.backward(dZ[-1].T)
        dH = dH.T
        dW.append((1/m) * dH.dot(x.T))
        dB.append((1/m) * dH.sum(axis = 1).reshape(-1, 1))

        return dZ, dW, dB, normalisation_layer

    """
    Calculate activation function
    """
    def rectifiedLinearUnit(self, x):
        # will return 0 if x < 0, otherwise x
        # ReLU activation
        return np.maximum(0, x)
        # alternative code for leaky ReLU
        # solves 'dying ReLU' problem
        return np.maximum(0.01 * x, x)

    """
    Calculate derivative of activation function
    """
    def derivative_ReLU(self, x):
        return (x > 0)
        # alternative code for leaky ReLU derivative
        return np.where(x > 0, 1, 0.01) 


    """
    Uses the normalised exponent function so the output can be show as a probability summing up to 1 
    this function is also known as softmax
    """
    def normalisedExpFunc(self, x):
        # first we need to do e^x for every value in x
        # then we get the sum of each 
        # then we finnally divide
        # clip values to stop exploding/vanishing gradients
        x = np.clip(x,-10,20)
        exp = np.exp(x - np.max(x))
        return exp / exp.sum(axis=0, keepdims=True)

    """
    Original update function
    depricated, replaced with rms prop
    """
    def update_parameters(self, a, weights, biases, dW, dB):
        # reverse deviations so layers line up
        dW = dW[::-1]
        dB = dB[::-1]
        for i in range(len(weights)):
            # subtract deviation from weights, multiply by alpha to scale this effect
            weights[i] = weights[i] - (a * np.clip(dW[i], -1000, 1000))
            biases[i] = biases[i] - (a * np.clip(dB[i], -1000, 1000))
        return weights, biases
    
    """
    Using RMS prop (Root Mean Square Propagation) to adaptively update parameters
    """
    def RMS_prop(self, g, a, lambda_reg, dW, dB, vdW, vdB, weights, biases, BatchNormLayer):
        # stored from last to first
        # flipped for ease of use
        dW = dW[::-1]
        dB = dB[::-1]

        # update gamma and beta used in batch normalisation
        # allows the values of gamma and beta to be learnt
        BatchNormLayer.vdGamma = g * BatchNormLayer.vdGamma + (1 - g) * BatchNormLayer.vdGamma ** 2
        BatchNormLayer.vdBeta = g * BatchNormLayer.vdBeta + (1 - g) * BatchNormLayer.vdBeta ** 2

        BatchNormLayer.gamma = BatchNormLayer.gamma - (a * BatchNormLayer.dGamma / np.sqrt(BatchNormLayer.vdGamma + 1e-10))
        BatchNormLayer.beta = BatchNormLayer.beta - (a * BatchNormLayer.dBeta / np.sqrt(BatchNormLayer.vdBeta + 1e-10))
        
        for i in range(len(weights)):
            # vdW and vdB refer to the moving average of squared gradients
            # They are updated using exponential decay (g)
            vdW[i] = g * vdW[i] + (1 - g) * np.sum(dW[i] ** 2)
            vdB[i] = g * vdB[i] + (1 - g) * np.sum(dB[i] ** 2)

            # Adjust weights and biases accordingly
            # The weights have also been regularised using L2 weight regularization
            # lamda_reg is the parameter for l2 weight regularization
            #weights[i] = weights[i] - (a * dW[i] / np.sqrt(vdW[i] + 1e-10)) - a * lambda_reg * weights[i]
            # L2 weight regularization causes worse performance so it has been disabled
            weights[i] = weights[i] - (a * dW[i] / np.sqrt(vdW[i] + 1e-10))
            biases[i] = biases[i] - (a * dB[i] / np.sqrt(vdB[i] + 1e-10))

        return weights, biases, vdW, vdB, BatchNormLayer

    """
    Train the network
    CHANGE skip_training to load data!!!
    """
    def train(self, train_data, train_labels, config, iterations, alpha, read_network, save_network):
        # Change skip_training to True to get pre defined weights and biases
        if read_network:
            self.read_from_file()
            return
        
        # Transpose training data for ease of use
        # This was initially easier but ended up being a pain with batch normalisation
        # Still need to do it as most of my system depends on this setup
        train_data = train_data.T
       
        # Initialse paramaters
        # gamma is the decay rate for RMS prop
        gamma = 0.99
        # lamda_reg is used for L2 weight regularization
        lambda_reg = 0.01

        # initalise weights and biases with given configuration
        weights, biases = self.initialise_parameters(config)
        # initialise moving average of squared gradients for RMS prop
        # same shape as weights and biases
        vdW = [np.zeros_like(weights[i]) for i in range(len(weights))]
        vdB = [np.zeros_like(biases[i]) for i in range(len(biases))]

        # Initialise normalisation layer
        normalisation_layer = BatchNormalisationLayer(np.ones(config[1]), np.zeros(config[1]))
        # Mini batches of size 50, 50 training examples at a time
        batch_size = 25
        # how many batches we have
        # example with 800 examples and a batch size of 50 we would have 800/50 = 16 batches
        num_batches = train_data.shape[1] // batch_size

    
        for i in range(iterations):
            # running loss for each batch to get mean
            total_loss = 0
            for batch in range(num_batches):
                # get data for current batch
                start = batch * batch_size
                end = start + batch_size
                X_batch = train_data[:, start:end]
                y_batch = train_labels[start:end]

                print(X_batch.size)
                print(y_batch.size)
                # foward propagation
                Zarr, Aarr, normalisation_layer = self.foward_propagation(X_batch, weights, biases, normalisation_layer)
                # backwards proagation to learn parameters
                _, dW, dB, normalisation_layer = self.back_propagation(X_batch, y_batch, Zarr, Aarr, weights, normalisation_layer)
                # use RMS prop to update values
                weights, biases, vdW, vdB, normalisation_layer = self.RMS_prop(gamma, alpha, lambda_reg, dW, dB, vdW, vdB, weights, biases, normalisation_layer)

                # calculate loss on the batch
                # use one hot encoding to create table of values
                y_out = np.zeros((y_batch.size, y_batch.max() + 1))
                y_out[np.arange(y_batch.size), y_batch] = 1
                # calculate loss using multiclass cross entropy loss
                # note this works for binary cases as well
                loss = np.sum(-y_out * np.log(Aarr[-1].T + 1e-10)) / X_batch.shape[0]
                loss = np.mean(-np.sum(y_out * np.log(Aarr[-1].T + 1e-10)))
                total_loss += loss

            # after 1000 iterations (epochs) show loss
            if (i % 1000 == 0):
                # loss is average of all batches
                print(f"EPOCH:{i} ~~~ loss:{total_loss / num_batches}")
        
        # save data for weights, biases and the normalisation layer to make predictions with
        self.normalisation_layer = normalisation_layer
        self.weights = weights
        self.biases = biases

        # save to file
        if save_network:
            self.save_to_file(weights, biases, normalisation_layer)

    """
    Used in training
    Splits data into k batches and used k-1 to train and the other batch to test
    This allows you to see if the network is genralising well to the dataset
    """
    def cross_fold(self, train_data, train_labels, config, iterations, a):
        k = 5
        # skl library kFold implementation
        kf = KFold(n_splits=k)
        history = []

        fold = 1
        # Splt data using k-fold
        for train_i, test_i in kf.split(train_data):
            print(f"Training on fold {fold}")

            # Generate a print
            print('----------------------------------------------------------------------')
            print(f'Training for fold {fold} ...')

            # Set training data and test data
            X_train = train_data[train_i]
            X_test = train_data[test_i]
            y_train = train_labels[train_i]
            y_test = train_labels[test_i]

            # Train network
            self.train(X_train, y_train, config, iterations, a)
            # Make predications
            predictions = self.predict(X_test)
            # Store accuracy
            accuracy = np.count_nonzero(predictions == y_test)/y_test.shape[0]
            history.append(accuracy)

            print(history)
            print('----------------------------------------------------------------------')

            fold+= 1

    """
    Save weights, biases, and normalisation data to files
    Load them in to skip training
    """
    def save_to_file(self, weights, biases, normalisationLayer):
        with open('weights.pkl', 'wb') as f:
            pickle.dump(weights, f)

        with open('biases.pkl', 'wb') as f:
            pickle.dump(biases, f)

        with open('normalise.pkl', 'wb') as f:
            pickle.dump(normalisationLayer, f)

    """
    Read stored data for neural network from file
    """
    def read_from_file(self):
        with open('weights.pkl', 'rb') as f:
            self.weights = pickle.load(f)
        with open('biases.pkl', 'rb') as f:
            self.biases = pickle.load(f)
        with open('normalise.pkl', 'rb') as f:
            self.normalisation_layer = pickle.load(f)

    """
    Make a prediction using the network
    """
    def predict(self, test_data):
        # Only need the last layer of Aarr, the probabilites
        _, Aarr, _ = self.foward_propagation(test_data.T, self.weights, self.biases, self.normalisation_layer)
        return np.argmax(Aarr[-1], 0)