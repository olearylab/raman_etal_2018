# -*- coding: utf-8 -*-
"""
Makes class neurNet: a feedforward neural network (can be subclassed later to
 make different types of network.).
neurNet is made up of different neurLayer classes.


The class setTask is not used in the paper
It sets a task, for the neural network, which can be populated with external data
 or self-generated student-teacher/classification data
"""


import numpy as np
import copy as copy

def updateOptions(self, dict_to_update = None, re_initialise_=False):
    for k in dict_to_update:
        self.options[k] = dict_to_update[k]

    if re_initialise_ is not False:
        self.re_initialise()


class neurLayer:
    def __init__(self, layer_size, input_size,layer_number):
        self.options = {}
        self.options['activation_function_choice'] = 'sigmoid'
        self.options['layer_size'] = layer_size
        self.options['input_size'] = input_size
        self.options['weight_init_stdev'] = 4*(input_size**(-0.5))
        self.options['weight_init_mean'] = 0
        self.options['bias_value'] = -1
        self.options['learning_rate'] = 0.02 * (input_size**(-0.5))
        self.options['layer_number'] = layer_number
        self.re_initialise()

    def re_initialise(self):
        if self.options['activation_function_choice'] == 'sigmoid':
            self.activation_function = lambda x: 1/(1 + np.exp(-x))
            self.activation_inverse_derivative = lambda x: x*(1-x)
        elif self.options['activation_function_choice'] == 'tanh':
            self.activation_function = lambda x: np.tanh(x)
            self.activation_inverse_derivative = lambda x: 1-x**2
        elif self.options['activation_function_choice'] == 'identity':
            self.activation_function = lambda x: x
            self.activation_inverse_derivative = lambda x: np.zeros_like(x) + 1
        # Initialise weights: 12 is to normalise stdev to 1 ## Should this be root 12. YES IT SHOULD
        # Also multiplied by 4 to give a bit more variance
        self.synaptic_weights = np.sqrt(12)*self.options['weight_init_stdev'] \
        * (np.random.random((self.options['layer_size'],
                             self.options['input_size']+1)) - 1/2) \
        + self.options['weight_init_mean']
        self.learning_rate = self.options['learning_rate']

    def forwardPass(self,input_vec, store_activation=False):
        number_of_inputs = input_vec.shape[1]
        input_vec = np.squeeze(np.append([input_vec], [np.tile( \
                self.options['bias_value'],[1,number_of_inputs])],axis=1),axis=0)
        activation = self.activation_function(self.synaptic_weights.dot(input_vec))
        if store_activation != False :
            self.activation = activation
            self.input_activation = input_vec
        return activation

    def backwardPass(self, output_vec, change_weights = False):
        #output_vec is the dEdo
        dEdoi = output_vec*self.activation_inverse_derivative(self.activation)
        #dimensionality is number of output neurons
        dEdin = np.dot(dEdoi.transpose(), self.synaptic_weights).transpose()
        dEdin = dEdin[0:-1, :] #get rid of bias neuron
        dEdw = np.dot(dEdoi,self.input_activation.transpose())
        if change_weights != False:
            self.synaptic_weights -= self.learning_rate * dEdw
        return dEdin, dEdw


class neurNet:
    def __init__(self, layer_size_vec):
        self.layers = []
        self.layer_size_vec = layer_size_vec
        self.current_time = 0
        self.re_initialise()
        self.num_layers = len(self.layer_size_vec)

    def re_initialise(self):
        self.layers = []
        self.all_synaptic_weights = []
        for i in range(1, len(self.layer_size_vec)):
            self.layers.append(neurLayer(self.layer_size_vec[i],
                                         self.layer_size_vec[i-1],i-1))
            self.all_synaptic_weights.append(self.layers[i-1].synaptic_weights)

    def forward_pass(self,input_vec,store_activation=False):
       #ugly compatibility function: implements evalNet exactly
       return self.evalNet(input_vec,store_activation=store_activation)

    def evalNet(self, input_vec,store_activation=False):
        # each column is a different input
        pass_particle = input_vec
        for i in range(len(self.layers)):
            pass_particle = self.layers[i].forwardPass(pass_particle,
                                       store_activation = store_activation)
        return pass_particle

    def getGradient(self, output_vec,label, change_weights=False):
        dEdo = output_vec - label
        target_ = dEdo
        gradient = []
        for i in range(len(self.layers)-1,-1,-1):
            target_, gradient_for_layer = self.layers[i].backwardPass(target_,
                                                     change_weights=change_weights)
            gradient.append(gradient_for_layer)
        gradient.reverse()
        return target_, gradient

    def trainNet(self, input_matrix = None, label_matrix = None,
                 number_of_runs = 100, task = None):
        # each column of the matrices should correspond to a different input/output
        if task is not None:
            if input_matrix is not None:
                print('Overwriting provided data with task data')
            input_matrix = task.training_inputs
            label_matrix = task.training_labels
        for i in range(number_of_runs):
            self.get_grad_from_input(input_matrix=input_matrix,
                                     label_matrix=label_matrix,
                                     change_weights=True)

    def get_num_weights(self):
        def layer_i_size(i):
            return self.all_synaptic_weights[i].size
        return np.sum(layer_i_size(i) for i in range(self.num_layers-1))

    def get_norm(self, weight_in):
        # Takes in weight-shaped object, and gets norm
        def get_each_norm(i):
            return np.linalg.norm(weight_in[i])
        norm_ =  np.sum(get_each_norm(i) for i in range(self.num_layers-1))
        if norm_ < 1e-10:
            norm_ = 0
        return norm_

    def normalise_weights_in(self, weight_in, scalar_to_mult=1):
        """
        Need to do for rd project. takes in weight-shaped object, and
        normalises and multiplies by scalar_to_mult. Assumes mutable object so
        no return value required.
        """
        norm_tot = self.get_norm(weight_in)
        if norm_tot is 0:
            mult_by = 0
        else:
            mult_by = (scalar_to_mult/norm_tot)
        for i in np.arange(self.num_layers-1):
            weight_in[i] *= mult_by

    def make_noise(self):
        return [np.random.randn(*self.all_synaptic_weights[i].shape) \
                for i in np.arange(self.num_layers-1)]

    def add_to_weights(self,direction, minus_='off'):
        #changes the synaptic weights by an amount direction
        for i in np.arange(self.num_layers-1):
            if minus_ is 'on':
                self.layers[i].synaptic_weights -= direction[i]
            elif minus_ is 'off':
                self.layers[i].synaptic_weights += direction[i]

    def scale_weights_in(self, weight_in, scalar_to_mult):
        """
        takes in weight shaped object and multiplies by scalar_to_mult
        """
        for i in np.arange(self.num_layers-1):
            weight_in[i] *= scalar_to_mult

    def sum_weights(self,*argv):
        #takes in separate lists of ndarrays. need to sum each one
        def sum_layer(i):
            return np.sum(arg[i] for arg in argv)
        return [sum_layer(i) for i in np.arange(self.num_layers-1)]

    def dot_weights(self, weights_1, weights_2):
        #takes in separate lists of ndarrays. need to sum each one. get \
        #overall dot product
        def i_dot(i):
            return np.sum(weights_1[i]*weights_2[i])
        return np.sum((i_dot(i) for i in np.arange(self.num_layers-1) ))

    def get_g(self, input_matrix, label_matrix, direction=None,
              fd_step = 0.0001, gradient_here=None,
              gradient_norm_here=None):
        fd_step=np.asarray(fd_step)


        dir_norm = self.get_norm(direction)

        self.scale_weights_in(direction, fd_step) # have to undo after



        #2. get grad (F(w+fd_step*dw)) by perturbing w with scaled direction
        self.add_to_weights(direction)
        grad_f_w_plus_delta_w = self.get_grad_from_input(input_matrix,
                                                          label_matrix)
        #3. undo perturbation to w
        self.add_to_weights(direction, minus_='on')

        #4. get gradF(w+fd_step*direction) - gradF(w)
        there_minus_here = [grad_f_w_plus_delta_w[i] - gradient_here[i] \
                              for i in np.arange(self.num_layers-1)]

        #5. Undo scaling of direction
        self.scale_weights_in(direction, (1/fd_step))

        #6. get <direction, there-here>
        inner_bracket_prod = self.dot_weights(there_minus_here, direction)

        #7. prepare scaling constant:
        sc_const = np.asarray(1/(2*gradient_norm_here*(dir_norm**2)*fd_step))
        return sc_const*inner_bracket_prod



    def get_grad_from_input(self,
                            input_matrix=None,
                            label_matrix=None,
                            change_weights=False):
        output_matrix = self.evalNet(input_matrix,store_activation=True)
        target, gradient = self.getGradient(output_matrix,
                                            label_matrix,
                                            change_weights=change_weights)
        return gradient

    def get_error(self, input_matrix, label_matrix):
        # Take in neuralNet class member, output L2 error
        output_matrix = self.evalNet(input_matrix,store_activation=False)
        return 0.5*np.sum( (output_matrix-label_matrix)**2)

class setTask:
    # This is not used in the paper: . We use a more asymmetric task given at the top of
    #pieces_new_resub_asymmetric.py. OR student teacher tasks given in pieces_new_resub
    def __init__(self,neurNet_ = None, input_dim = None, output_dim = None,
                 task_type = 'student-teacher', online_generation = False,
                 external_training_inputs = None, external_test_inputs = None,
                 external_training_labels = None, external_test_labels = None,
                 training_size = 2000, test_size = 500,
                 teacher_network = None):
        self.options = {}
        self.all_task_types = {'student-teacher' : 1, 'classification' : 2, 'external_data' : 3}
        self.options['task_type'] = task_type
        self.task_type_local = self.all_task_types.get(self.options['task_type'])
        self.teacher_network = teacher_network
        self.options['training_size'] = training_size
        self.options['test_size'] = test_size
        self.options['online_generation'] = online_generation
        self.online_generation = self.options['online_generation']
        if online_generation is not False:
            self.options['training_size'] = None
        if self.task_type_local == 1:
            if neurNet_ is None and teacher_network is None:
                print('Need either a teacher_network or neurNet to copy \
                      for teacher generation. Aborting task construction')
            elif teacher_network is None and neurNet_ is not None:
                self.teacher_network = neurNet(neurNet_.layer_size_vec)
        if neurNet_ is not None:
            self.input_dimension = neurNet_.layers[0].options['input_size']
            self.output_dimension = neurNet_.layers[-1].options['layer_size']
        else:
            self.input_dimension = input_dim
            self.output_dimension = output_dim
        if external_training_inputs is not None:
            self.options['online_generation'] = False
            self.options['task_type'] = 'external_data'
            self.training_inputs = external_training_inputs
            self.test_inputs = external_test_inputs
            self.training_labels = external_training_labels
            self.test_labels = external_test_labels
            self.input_dimension = external_training_inputs.shape[0]
            self.output_dimension = external_training_labels.shape[0]
            self.options['training_size'] = external_training_inputs.shape[1]
            self.options['test_size'] = external_test_inputs.shape[1]
        else:
            self.populate_data()

    def re_initialise(self):
        print('The setTask options cannot be updated. \
              This operation has corrupted the task object. Please set a new task')

    def grab_data(self, data_cardinality=100):
        # grab data. if online_generation is on, then this passes to generate_data.
        # if online_generation is off, this grabs randomly from pre-loaded data
        if self.online_generation is True:
            input_matrix, label_matrix = generate_data(data_cardinality = data_cardinality,
                                                       my_task = self.task_type_local)
        else:
             total_num_data = self.training_inputs.shape[1]
             to_choose = np.random.choice(total_num_data,data_cardinality)
             input_matrix, label_matrix = self.training_inputs[:,to_choose],
             self.training_labels[:,to_choose]
             # NOT using replace = false for random sample of training data
        return input_matrix, label_matrix

    def generate_data(self, data_cardinality = None, my_task = None):
        # generate data according to task type. if external, then do not use.
        # if online_generation is off, do not use after initialisation.
        if my_task is 1:
            input_matrix = np.random.random([self.input_dimension, data_cardinality])-0.5
            label_matrix = self.teacher_network.evalNet(input_matrix)
        elif my_task is 2:
            input_matrix, label_matrix \
            = make_classification_task_data(data_cardinality = data_cardinality,
                                            input_dimension = self.input_dimension,
                                            output_dimension=self.output_dimension)
        return input_matrix, label_matrix

    def populate_data(self):
        # only use if external data is off. In this case, populate test data/labels. \
        # If online_generation is off, additionally populate training data/labels
        if self.task_type_local is 3:
            print('External data, cannot repopulate')
            return
        training_inputs_, training_labels_ \
        = self.generate_data(data_cardinality = self.options['training_size'], my_task = self.task_type_local)
        if self.online_generation is False:
            self.training_inputs, self.training_labels = training_inputs_, training_labels_
        elif self.online_generation is True:
            self.test_inputs, self.test_labels = training_inputs_, training_labels_
        # finish once generate_data is done: want test_inputs to be a subset of training data for online_generation off

def make_classification_task_data(data_cardinality = 100, input_dimension = 10, output_dimension = 3,sorted = False):
    # This is not used in the paper. We use a more asymmetric task given at the top of
    #pieces_new_resub_asymmetric.py
    # Makes output_dimension number of random reference vectors.\
    # Generates/classifies inputs dependent on dot products with reference
    reference_vecs = 2*input_dimension*np.random.random([input_dimension,
                                                         output_dimension]) - input_dimension
    input_matrix = np.random.random([input_dimension, data_cardinality]) - 0.5 \
    + 0.05 * np.squeeze(reference_vecs[:,[np.random.randint(0,output_dimension,
                                                            size=data_cardinality)]])
    label_ref = np.dot(reference_vecs.transpose(),input_matrix).transpose().argmax(axis=1).transpose()
    # since matrix logic is faster in axis=-1
    label_matrix = np.zeros([output_dimension,data_cardinality])
    label_matrix[label_ref,np.arange(0,data_cardinality)] = 1
    if sorted is True:
        label_order = label_ref.argsort()
        input_matrix = input_matrix[:,label_order]
        label_matrix = label_matrix[:,label_order]
    return input_matrix, label_matrix

class weight_size:
    # Makes a class that is a list of ndarrays TO DO!!!!!!!!!!!
    def __init__(self,size_vec):

        for i in np.arange(len(size_vec)):
            print('hi')
