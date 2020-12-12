#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ----------Jiaqi Chen-------------------
#----------Fianl project------------------
# Code for a 3-layer neural network, and code for learning the MNIST dataset

from time import time
import numpy 
# scipy.special for the sigmoid function expit()
import scipy.special
import matplotlib.pyplot
# ensure the plots are inside this jupyter notebook, not an external window
get_ipython().run_line_magic('matplotlib', 'inline')
# helper to load data from PNG image files
import imageio
# glob helps select multiple files using patterns
import glob


# In[2]:


# neural network class definition （3 layers）
class neuralNetwork:
    # initialise the neural network
    def __init__(self,inputnodes,hiddennodes,outputnodes,learningrate):
        # set number of nodes in each input,hidden,output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # learning rate
        self.lr = learningrate
        
        # link weight matrices ,wih and who
        # weithg inside the arrays are w_i_j, where link is from node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = (numpy.random.normal(0.0, pow(self.hnodes,-0.5), (self.hnodes,self.inodes) )  )
        self.who = (numpy.random.normal(0.0, pow(self.onodes,-0.5), (self.onodes,self.hnodes) )  )
        
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
 
        pass
    
    # train the neural network
    def train(self,inputs_list,targets_list):
        # convert inputs list to 2d array        
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        # output layer error is the (target-actual)
        output_errors = targets - final_outputs
        # hidden layer error is the output_errors,split by weights,recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        
        pass
    
    # query the neural network
    def query(self,inputs_list):
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list,ndmin=2).T
        
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        
        return final_outputs


# In[3]:


# number of input,hidden and output nodes
# 28 * 28 = 784
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
 
# learning rate is 0.x
learning_rate = 0.2
 
# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes,learning_rate)
 
# train the neural network
 
# load the mnist training data csv file into a list
training_data_file = open("mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
training_data_list  = training_data_list[:10000]
# epochs is the number of times the training data set is used for training


# In[4]:


epochs = 9
print("Train...")
start = time()

for e in range(epochs):
    # go through all records in the training data set
    for record in training_data_list:
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 ) 
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) 
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 1
        n.train(inputs,targets)
        pass
    pass

end = time()
t = end - start
print('Train：%dmin%.3fsec' % (t//60, t - 60 * (t//60)))


# In[5]:


#np.shape(inputs)


# In[6]:


#np.shape(targets)


# In[7]:


#np.shape(training_data_list)


# In[8]:


# test the neural network
 
# load the mnist test data csv file to a list
test_data_file = open("mnist_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
test_data_list  = test_data_list[:1000]
# scorecard for how well the network performs,initially empty 
scorecard = []
# go through all records in the test data set
for record in test_data_list:
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0) 
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
#    print("Answer label is:",correct_label," ; ",label," is network's answer")
    # append correct or incorrect to list
    if(label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        scorecard.append(0)        
    pass
 
# calculate the performance score ,the fraction of correct answers
scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size )


# In[ ]:




