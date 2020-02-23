import numpy as np
import random

class Network(object):
    ## size - number of neurons in the respective layer of network
    ## For example is list was [2 3 1] => net = Network([2, 3, 1])
    ## That means its a 3 layered network with first layer.
    ## containing 2 neurons, second layer containing 3 neurons and 3rd layer containing 1 neuron
    ## Weights and biases are declared randomly initially using gausian distribution with mean 0 and variance 1
    ## 1st layer is input layer so there won't be any biases for this layer
    ## np.random.rand(3,2)  - creates an array of 3 by 2 random numbers
    def __init__(self,sizes): 
      # print(sizes)
        self.numberOfLayer = len(sizes)
      # print(self.numberOfLayer)
        self.sizes = sizes
        self.biases = [np.random.randn(i,1) for i in sizes[1:]] # b1 = 3 by 1 | b2 = 1 by 1
       # print(self.biases)

        self.weights = [np.random.randn(j,i)  for i,j in zip(sizes[0:], sizes[1:])] # w1 = 3 by 2 | w2 = 1 by 3
       # print(self.weights)
        ## Python's zip function creates an iterator that aggregates elements from 2 or more iterables
        ## here zip creates a common iterator for the different elements in different layers
        ## example ->
        ## numbers = [1, 2, 3]
        ## >>> letters = ['a', 'b', 'c']
        ## >>> zipped = zip(numbers, letters)

    def feedForward(self,a):
        ## a is the  input to the network initially 
        ## after that a returned by this funciton is the output of the network
        for b,w in zip(self.biases,self.weights):
            #print(b)
            #print(w)
            z = np.dot(w, a) + b
            a = sigmoid(z)
            #print(a)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        ## we will be training our network using mini batch stochastic gradient descent
        ## training_data is a list of tuples x,y representing training inputs and outputs
        ## If test_data is provided then the network will be evaluated against test data after each epoch, and
        ## partial progress will be printed out . This u=is useful to track progress, but slows things down
        ## eta - is learning rate
        if test_data: n_test = len(test_data)
        ## difference between range and xrange - range returns the list of numbers
        # xrange returns the generator object and to print the numbers in this object we need touse loop
        # BUT IN PYTHON 3 THERE IS NO xrange function so we are using range only
        n = len(training_data)
        for j in range(epochs):
            ## step 1: shuffle the traininf data
            random.shuffle(training_data)
            ## step 2: partition the data into mini batches / divide whole bunch of mini batches
            mini_batches = [ training_data[k:k+mini_batch_size]  for k in range(0,n,mini_batch_size)]
            ## step 3: then for each mini batch we apply single iteration of gradient descent
            ##         this is done by the function self.update_mini_batch(mini_batch,eta), which update
            ##         network weights and biases according to single iteration of gradient descent, using the training data in mini_batch
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch,eta)

            if test_data:
                print ("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print ("Epoch {0} complete!!".format(j))

    def update_mini_batch(self, mini_batch,eta):
        ## this code updates the networks weights and biases by applying gradient descent using backpropogation to single mini batch.
        ## mini_batch is the list of tuple X and y 

        ## eta - learning rate
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x,y in mini_batch:
           delta_nabla_b,delta_nabla_w = self.back_prop(x,y) ## getting small adjustments in weights and biases using gradient descent
                                                             ## for a particular mini batch
           nabla_b = [nb+dnb for nb, dnb in zip(nabla_b,delta_nabla_b)] ## adding small change in bias to the initialized bias
           nabla_w = [nw+dnw for nw, dnw in zip(nabla_w,delta_nabla_w)] ## adding small chnage in weights to the initialized weight

        self.biases = [ b - (eta/len(mini_batch)) * nb for b,nb in zip(self.biases,nabla_b)] # updates weights nd biases for all examples together in mini batch 
        self.weights = [ w - (eta/len(mini_batch)) * nw for w,nw in zip(self.weights,nabla_w)]
    
    def cost_derivative(self,output_activations,y):
        return (output_activations-y)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedForward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def back_prop(self,x,y):
        ## now the purpise of backprop is to use compute the small changes that we need to make to our weights and biases
        ## to get the least cost - which we can get using concept of gradient descent
        nabla_b = [ np.zeros(b.shape) for b in self.biases]
        nabla_w = [ np.zeros(w.shape) for w in self.weights]

        ## use feed forward to get the output of network for this particular minimat
        a = x
        activations = [x] ## list that stores all activations layer by layer
        zs = [] ## list to store all z vectors layer by layer
        for b,w in zip(self.biases,self.weights):
            z = np.dot(w, a) + b
            zs.append(z)
            a = sigmoid(z)
            activations.append(a)

        ## now use backward pass to compute small change in b and w

        ## delta here represent derivative of cost wrt bias
        delta = self.cost_derivative(activations[-1], y) * sigmoid_derivative(zs[-1]) # we are calulating it for last layer
        ## dC/db(L) = ( dZ(L)/db(L) ) * ( da(L)/dz(L) ) * ( dc/da(L) ) = 1 . sigmoid_derivative(z(L)) . 2 * ( a(L) - y )
        ## dC/db(L) = ( dZ(L)/dw(L) ) * ( da(L)/dz(L) ) * ( dc/da(L) ) = a(L-1) . sigmoid_derivative(z(L)) . 2 * ( a(L) - y )
        ##                                                             = a(L-1) . dC/db(L)
        ##                                                             = a(L-1) . delta
        nabla_b[-1] = delta ## calulated chnages in bias and weights for last layer in neural network
        nabla_w[-1] = np.dot(delta,activations[-2].transpose())

        ## now we have calculate it for whole network i.e., each layer in network

        for l in range(2, self.numberOfLayer):
            z = zs[-l]
            sd = sigmoid_derivative(z)
            ## now for the 2nd layer from the end we have to calculate new delta or new change in bias 
            ## for that we multiple the weights used for last layer with delta of previous layer (chnage in bias of previous layer)
            ## this process continues until we reach first layer
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sd  #  ( w(L) .delta ) * sigmoid_derivative(Z(L-1))
                                                                     # doing this we calculate the change in bias for 2nd layer from end
                                                                     # if we multiply the rate of change in cost wrt to baises for last layer
                                                                    # with layer next to last then we can calcluate change due actiavtion of that layer
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        ## in the end we return all the chnages that are necessary in weights and biases for the particular mini batch
        return (nabla_b, nabla_w)


     

        

def sigmoid(z):
    return 1.0/(1.0 + np.exp(-z))
def sigmoid_derivative(z):
    return sigmoid(z) * (1-sigmoid(z))
    
##net = Network([2,3,1])
##print(net.feedForward([2 ,3]))