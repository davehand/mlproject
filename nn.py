'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

class NeuralNet:

    def __init__(self, layers, learningRate, epsilon=0.12, numEpochs=100):
        '''
        Constructor
        Arguments:
        	layers - a numpy array of L-2 integers (L is # layers in the network)
        	epsilon - one half the interval around zero for setting the initial weights
        	learningRate - the learning rate for backpropagation
        	numEpochs - the number of epochs to run during training
        '''
        self.layers = layers
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.numEpochs = numEpochs
        self.theta = {}
        self.a = []
        self.num_layers = (len(self.layers) + 2)
        self.regLambda = .000001 # 1e-8
        self.hasConverged = False
        self.cost = 1000


    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoidGradient(self, z):
        return np.multiply(self.sigmoid(z), (1 - self.sigmoid(z)))

    def computeCost(self, a, y, n):

        Y = label_binarize(y, classes=np.unique(y))
        # Y = y

        # print -(1.0/n) * (np.sum( np.dot(Y.T,np.log(a)) + np.dot((1-Y).T,np.log(1-a)) ) )
        errors = np.zeros((n,1))
        for i in range(0, n):
            errors[i] =  np.sum( Y[i] * np.log(a[i]) + (1-Y[i]) * np.log(1-a[i]) )
        
        # theta = self.theta.copy()
        theta = 0
        for i in range(0, len(self.theta)):
            myTheta = self.theta[i]
            theta += np.sum(myTheta[:,1:]**2)

        J = -(1.0/n) * np.sum(errors)
        J = J + (self.regLambda/(2*n) * theta)
        

        return J


    def unrollTheta(self):
        
        theta = []
        # print theta.shape

        for k, v in self.theta.iteritems():
            if type(v) is np.ndarray:
                for t,c in enumerate(v):
                    for item in c:
                        theta.append(item)
        
        return theta
    
    def computeGradient(self, grad, theta, n):

        
        num_layers = len(grad)

        J = [0] * num_layers
        myTheta = theta.values()
        # print grad[0],'\n\n'
        # print grad[1]
        # i = 0
        # for j in range(num_layers-1, -1, -1):
        for j in range(0, num_layers):
            
            d1 = (grad[j] / n)
            v = np.copy(myTheta[j])
            # tmp = np.zeros((v.shape[0],1))
            tmp = np.zeros((1,v.shape[1]))

            # print tmp.shape
            # print v.shape
            v[0] = tmp
            # print v
            # v[:,0] = tmp.T
            
            # print np.multiply(self.regLambda, v)
            
            J[j] = d1 + np.multiply(self.regLambda, v)  # regularization
            
            # print J[j].shape
        
        return J

    

    def back_propagation(self, X, y, theta):

        delta = {}
        n,d = X.shape
        layer_count = len(self.theta) 
        grad = [0] * layer_count 
        
        I = np.eye(np.unique(y).shape[0])
        A = np.zeros((n, np.unique(y).shape[0]))
        
        for i in range(0, n):
            a = self.forward_propagation(X[i])
            delta[layer_count] = a[-1] - I[y[i]] # d3
            # print delta[layer_count]
            for j in range(layer_count-1,0,-1):
                # print j
                thet = theta[j]
                # thet = thet[:,1:]
                # print thet
                delta[j] = np.multiply(np.dot(thet.T, delta[(j+1)]), self.sigmoidGradient(a[j]))
                delta[j] = delta[j][1:]
                # print delta[j].shape  tablesgenerator.com
            
            g = 0
            for p in range(layer_count-1, -1, -1):
                grad[p] += np.dot(delta[p+1].reshape((-1,1)) , a[p].reshape((-1,1)).T)
                # grad[p] = grad[p][:,1:]
                # print grad[p].shape
                g += 1
        
            # print grad
            A[i] = a[-1]
            

        self.cost = self.computeCost(A, y, n)

        return self.computeGradient(grad, theta, n)


    def gradientCheck(self):
        I = np.eye(len(self.theta))
        print I
        # gradApprox = 
        return I

    def forward_propagation(self, X):


        n = len(self.theta)
        self.a = [0]* self.num_layers
        # self.a[0] = np.concatenate((np.ones((X.shape[0],1)), X), axis=1).T
        self.a[0] = np.concatenate(([1], X), axis=1).T
        # print layers[0].shape
        theta = self.theta
        
        for i in range(0, len(theta)-1):
            a = self.sigmoid(np.dot(theta[i], self.a[i]))
            # a1 = np.concatenate((np.ones((1, a.shape[1])), a), axis=0)
            # a1 = np.concatenate(([1], a), axis=1).T
            
            self.a[(i+1)] = np.insert(a, 0, [1])
        
        i += 1
        self.a[(i+1)] = (self.sigmoid(np.dot(theta[i], self.a[i])))
        

        return self.a
    
    def forward_propagation_all(self, X):

        n = len(self.theta)
        self.a = [0]* self.num_layers
        self.a[0] = np.concatenate((np.ones((X.shape[0],1)), X), axis=1).T
        theta = self.theta
        
        for i in range(0, len(theta)-1):
            a = self.sigmoid(np.dot(theta[i], self.a[i]))
            a1 = np.concatenate((np.ones((1, a.shape[1])), a), axis=0)
            self.a[(i+1)] = a1
        
        i += 1
        self.a[(i+1)] = (self.sigmoid(np.dot(theta[i], self.a[i])))
        
        return self.a

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        n,d = X.shape
        # print n,d
        output_values = np.unique(y) # K are output units
        K = len(output_values)
        
        output_layer_sz = d + 1
        for i,c in enumerate(self.layers):
            input_layer_sz = self.layers[i]
            # print input_layer_sz, output_layer_sz
            self.theta[i] = np.random.uniform(-self.epsilon, self.epsilon, (input_layer_sz, output_layer_sz))
            output_layer_sz = input_layer_sz + 1
            
        input_layer_sz = K
        i += 1
        
        # print input_layer_sz, output_layer_sz

        self.theta[i] = np.random.uniform(-self.epsilon, self.epsilon, (input_layer_sz, output_layer_sz))
        # print self.theta
        # theta1 = np.reshape(self.theta[0:100], (10,10))
        
        theta = self.theta.values()
        
        D = {}
        

        last = 0
        for i in range(0,self.numEpochs):
            grad = self.back_propagation(X, y, self.theta)
            # print grad
            for j in range(0,len(grad)):
                D[j] = (self.learningRate * np.array(grad[j]))
                # print D[j]
                self.theta[j] -= D[j]

            if self.cost == last:
                # print 'Converged.'
                break

            last = self.cost
            # print last
            

            # self.theta = theta
            # print self.theta




    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        n,d = X.shape
        # print n,d
        
        h = np.zeros((n,len(self.a[-1])))

        for i in range(0,n):
            h[i] = self.forward_propagation(X[i])[-1]


        # print h
        
        return np.argmax(h, axis=1)
    
    
    def visualizeHiddenNodes(self, filename):
        '''
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''
        theta = self.theta[0][:,1:]
        n,d = theta.shape
        width = int(np.round(np.sqrt(d)))
        height = int((d / width))
        rows = int(np.floor(np.sqrt(n)))
        cols = int(np.ceil(n / rows))

        padding = 1

        pixels = np.ones((padding + rows * (height + padding), padding + cols * (width + padding)))

        example = 0
        for i in range(1, rows+1):
            for j in range(1, cols+1):
                if example > n:
                    break

                max_intensity = np.max(np.abs(theta[example, :]))
                y_start = np.add(padding + (i-1) * (height + padding), 0)
                y_end = np.add(padding + (i-1) * (height + padding), height)
                x_start = np.add(padding + (j-1) * (width +  padding), 0)
                x_end = np.add(padding + (j-1) * (width +  padding), width)
                pixels[y_start:y_end,x_start:x_end] = theta[example].reshape((height, width)) / max_intensity
                example += 1
            if example > n:
                break

        
        plt.figure(1)
        plt.imshow(pixels, cmap=plt.cm.gray_r, interpolation='nearest')
        plt.axis('off')
        plt.show()

        plt.imsave(filename, pixels, cmap=plt.cm.gray_r, format='png')


