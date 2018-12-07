#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 10:08:13 2018
Replaces pieces. Makes family of neural nets.
@author: dvr23
"""
import abc
import numpy as np
import copy as copy
import setupMLP as s
import matplotlib.pyplot as plt



class nn_family:
    def __init__(self, num_nets=None,
                 base_dim=None,
                 data_cardinality=None):
        self.nn_index = list()
        self.data_cardinality = data_cardinality
        self.base_dim=base_dim
    """
    def create_data(self):
        print('hi')
    """

    def add_nn(self,nn_in):
        """
        The subclass adds an NN subject to the constraints
        imposed by the i-o data. This superclass method adds
        it to a dictionary which indexes the NNs, Also adds input/output list
        corresponding to i-o data for each NN. (for MLP these will be
        identical)
        """
        self.nn_index.append(nn_in)

    def set_which_teacher(self, input_dict):
        ## for ith nn_index, which teacher should i train on? key is nn_index
        #, value is which teacher
        self.which_teacher_dict.update(input_dict)


    def get_ith_io(self,i_val,is_online):
        """
        Gets inputs and outputs for ith network.
        """
    def get_n_vec(self,i_vec):
        def ith_n(i):
            return self.nn_index[i].get_num_weights()
        return np.asarray([ith_n(i) for i in i_vec])


    def expected_steady_state_G(self, gamma=None, dt=None,
                                gamma_info=None, N_vec=None):
        # Get G according to paper formula for a given
        # network. Needs number of weights of network, gamma, and dt
        """
        Take expression for learning rate. Check what G has to be for learning rate to equal zero
        Equality to solve: G = gamma[0] / ( \|dot{w}\|_2^2 dt )
        dot{w}^2 = gamma_1^2 + gamma_2^2 without intrinsic noise
        dot{w}^2 = gamma_1^2 + gamma_2^2 + (N/dt)gamma_3^2 with intrinsic noise
        """
        if gamma_info is not None:
            delta_t = gamma_info['delta_t']
            gamma = gamma_info['gamma']

        dotw2 = gamma[0]**2 + gamma[1]**2  + (N_vec/delta_t)*gamma[2]**2
        return gamma[0]/(dotw2*delta_t)

    def fake_initialise(self, indices_to_run=None,
                         gamma_=None,
                         number_of_runs=1000,
                         delta_t=None,
                         gamma1il = [0.96, 0.04],
                         subsample_fraction=1):
        """ Do a training run on a fake teacher to mix up the initial weights
        into a similar distribution as they will be after training on the real
        teacher. Call this from training run automatically
        """

        #temp_outs = output_class(nn_family_=self)
        temp_outs = self.training_run(indices_to_run=indices_to_run, gamma_=gamma_, \
        number_of_runs = number_of_runs, delta_t = delta_t, gamma1il=gamma1il,
        subsample_fraction=subsample_fraction,
        is_online=0, real_teacher=0)

        #import ipdb; ipdb.set_trace()
        # modify weights appropriately
        for j in range(len(self.base_net)):
            if hasattr(self.base_net[j], 'layers'):
                for i,layer in enumerate(self.base_net[j].layers):
                    layer.synaptic_weights = temp_outs.final_weights[j][i]
            else:
                self.base_net[j].all_synaptic_weights = \
                temp_outs.final_weights[j]

        for net in self.nn_index:
            self.equalise_network_weights(net)

    def train_with_initialise(self, indices_to_run=None,
                     gamma_=None,
                     number_of_runs=None,
                     delta_t=None,
                     gamma1il = [0.96, 0.04], outs_class_in=None,
                     subsample_fraction=1,
                     is_online=0,
                     fake_runs=500):
         """
         Does a fake initialise and a training run with the same input parameters
         """
         
         
         
         self.fake_initialise(indices_to_run=None,
                              gamma_=gamma_,
                              number_of_runs=fake_runs,
                              delta_t=delta_t,
                              gamma1il=gamma1il,
                              subsample_fraction=subsample_fraction)


         nn_family.training_run(self,
                           indices_to_run=indices_to_run,
                           gamma_=gamma_,
                           number_of_runs=number_of_runs,
                           delta_t=delta_t,
                           gamma1il=gamma1il,
                           outs_class_in=outs_class_in,
                           subsample_fraction=subsample_fraction,
                           is_online=is_online)


    def training_run(self, indices_to_run=None,
                     gamma_=None,
                     number_of_runs=None,
                     delta_t=None,
                     gamma1il = [0.96, 0.04], outs_class_in=None,
                     subsample_fraction=1,is_online=0,real_teacher=1):


# =============================================================================
#         Systematic noise keeps gamma1il[1] of its old value each timestep, and
#         innovates gamma1il[0] of itself.
#         As we increase delta t, gamma1il[1] should decrease exponentially. We
#         use our_gamma1il to reflect this.
#       We will use real_teacher = 1 for the actual training, and real_teacher = 0
#       for the fake training. The latter trains the base net only. After this, all the student
#       networks can be weight-equalised with the fake-trained base net. This ensures agnostic
#       initialisation for the real training run.
# =============================================================================

        wt=real_teacher
        if wt is 1:
            # get_ith_io then comes from real teachers. nn_index are trained
            ind_num = range(len(indices_to_run))
            #indices_to_run=[0]
            nn_ind = copy.deepcopy([self.nn_index[i] for i in indices_to_run])
        elif wt is 0:
            #get_ith_io comes from fake teachers. base_nets are trained
            ind_num = range(len(self.base_net))
            indices_to_run = np.asarray(ind_num)
            nn_ind = [self.base_net[i] for i in ind_num]

        if outs_class_in is None:
            outs_class_in = output_class()
            outs_class_in.sizes_vec = [nn_ind[i].layer_size_vec  for i in ind_num]

        delta_t = np.asarray(delta_t)
        gamma_=np.asarray(gamma_)
        outs_class_in.gamma_info = {'gamma' : gamma_, 'delta_t' : delta_t}
        # multiply by dt to modify step size. gamma[2] later divided by sqrt(dt)
        our_gamma = gamma_*delta_t
        our_gamma1il = copy.deepcopy(np.asarray(gamma1il))
        our_gamma1il[1] = our_gamma1il[1]**delta_t
        outs_class_in.gamma1il = our_gamma1il[1]
        outs_class_in.n_vec = self.get_n_vec(indices_to_run)
        # Initialise variables tracking state of run
        error_list = np.zeros([number_of_runs, len(ind_num)])
        g_list = np.zeros([number_of_runs, len(ind_num)])
        grad_norm_list = np.zeros([number_of_runs, len(ind_num)])
        step_norm = np.zeros([number_of_runs, len(ind_num)])

        # Generate appropriate inputs and outputs:
        io_to_use= [self.get_ith_io(i,is_online,wt) \
                    for i in ind_num]
        #io_to_use = list(map(self.get_ith_io,ind_num))
        inputs_to_use = [io_to_use[i][0] for i in ind_num]
        labels_to_use = [io_to_use[i][1] for i in ind_num]
        # print(inputs_to_use[2] - self.get_ith_io(2)[0])
        num_inputs = io_to_use[0][0].shape[1]
        subsample_size= round(num_inputs*subsample_fraction)
        def dist_error(i):
            return nn_ind[i].get_error(inputs_to_use[i],
                         labels_to_use[i])

        def dist_gradient(i):
            return nn_ind[i].get_grad_from_input(
                    inputs_to_use[i][:,subsample_nums],
                    labels_to_use[i][:,subsample_nums])

        def dist_gradient_true(i):
            return nn_ind[i].get_grad_from_input(
                    inputs_to_use[i],
                    labels_to_use[i])

        def dist_get_norm(i, weights_in):
            return nn_ind[i].get_norm(weights_in[i])

        def dist_normalise_weights_in(i, weights_in, gamma_in):
            #nb weight shaped matrices not just weights. eg noise
            # Use dist_normalise_weights_in(i,list_of_weights,scaling_factor)
            nn_ind[i].normalise_weights_in(weights_in[i],
                  scalar_to_mult = gamma_in)

        def dist_scale_weights_in(i, weights_in, gamma_in):
            nn_ind[i].scale_weights_in(weights_in[i], gamma_in)

        def dist_make_noise(i):
            #everybody make some noise
            noise_now = nn_ind[i].make_noise()
            return noise_now

        def dist_sum(i,*summands):
            # use dist_sum(i, list1_to_sum[i],list2_to_sum[i])
            return nn_ind[i].sum_weights(*summands)

        def dist_get_g(i, direction_in, gradient_here=None,
                       gradient_norm_here=None):
            return nn_ind[i].get_g(inputs_to_use[i],
                         labels_to_use[i],
                         direction=direction_in[i],
                         gradient_here=gradient_here[i],
                         gradient_norm_here=gradient_norm_here[i])


        noiseg1_now = [dist_make_noise(i) for i in ind_num]
        any(dist_normalise_weights_in(i, noiseg1_now,
                                          our_gamma[1]) for i in ind_num)
        # run loop
        for time_ in np.arange(number_of_runs):
            if is_online is 1:
                io_to_use= [self.get_ith_io(i,is_online, \
                                            wt) for i in ind_num]
                inputs_to_use = [io_to_use[i][0] for i in ind_num]
                labels_to_use = [io_to_use[i][1] for i in ind_num]
            subsample_nums = np.random.choice(num_inputs, subsample_size,
                                              replace=False)
            error_list[time_,:] = [dist_error(i) for i in ind_num]
#           error_list[time_,:] = list(map(dist_error, indices_to_run)) #alternative
            gradients_now = [dist_gradient(i) for i in ind_num]
            if subsample_size < num_inputs:
                gradients_now_true = [dist_gradient_true(i) for i in ind_num]
            else:
                gradients_now_true = copy.deepcopy(gradients_now)
            gradient_norms = [dist_get_norm(i, gradients_now_true) \
                              for i in ind_num]
            grad_norm_list[time_,:] = np.copy(gradient_norms)



            # scale gradient by norm and gamma0
            #print(dist_normalise_weights_in(1, gradients_now,-our_gamma[0]))
            any(dist_normalise_weights_in(i, gradients_now,
                                          -our_gamma[0]) for i in ind_num)



# =============================================================================
#           innovation to systematic error. needs to be normalised to
#           ourgammil[0]/ourgammil[1] as fraction of existing systematic
#           term. we normalise first so that there is no division by zero in
#             case of the scaling our_gamm1il[1] = 0
# =============================================================================
            noiseg1_now_innov = [dist_make_noise(i) for i in ind_num]
            #existing noiseg1 should have norm our_gamma[1] at this point
            any(dist_normalise_weights_in(i, noiseg1_now_innov,
                                          our_gamma[1]) for i in ind_num)
            any(dist_scale_weights_in(i, noiseg1_now_innov,
                                          our_gamma1il[0]) for i in ind_num)

            any(dist_scale_weights_in(i, noiseg1_now,
                                          our_gamma1il[1]) for i in ind_num)

# =============================================================================
#             Sum innovation and existing terms for systematic error.
# =============================================================================

            noiseg1_now = [dist_sum(i, noiseg1_now[i],
                                   noiseg1_now_innov[i]) for i in ind_num]

            # make some noise: intrinsic
            noiseg2_now = [dist_make_noise(i) for i in ind_num]


            #normalise both sets of noises
            any(dist_normalise_weights_in(i, noiseg1_now,
                                          our_gamma[1]) for i in ind_num)
            any(dist_normalise_weights_in(i, noiseg2_now,
                                          np.sqrt(outs_class_in.n_vec[i]/ \
                                                  delta_t)*our_gamma[2]
                                          ) for i in ind_num)
            #################################################


            new_direction = [dist_sum(i, gradients_now[i],
                                      noiseg1_now[i],
                                      noiseg2_now[i]) for i in ind_num]
            step_norm[time_,:] = [dist_get_norm(i,new_direction) \
                     for i in ind_num]
            g_list[time_,:] = [dist_get_g(i, new_direction,
                  gradient_here=gradients_now_true,
                  gradient_norm_here=gradient_norms) for i in ind_num]

            #print(new_direction)
            #print(new_direction[1][1,1])
            #print(gradients_now[1][1,1] + noise1_now[1][1,1] + noise2_now[1][1,1])
            for i in ind_num:
                nn_ind[i].add_to_weights(new_direction[i])
        #outs_class_in.nn_index = nn_ind
        outs_class_in.error_list = error_list
        outs_class_in.g_list = g_list
        outs_class_in.grad_norm_list = grad_norm_list
        outs_class_in.step_norm = step_norm
        outs_class_in.final_weights = [nn_ind[i].all_synaptic_weights
                              for i in ind_num]
        outs_class_in.expected_g_vec = self.expected_steady_state_G(
                gamma_info=outs_class_in.gamma_info, N_vec=outs_class_in.n_vec)
        outs_class_in.is_online = is_online
        return outs_class_in

class nn_fam_linear(nn_family):
    def __init__(self, num_nets=None,
                 base_dim=None,
                 data_cardinality=None, num_teachers = 2):

        nn_family.__init__(self,num_nets=num_nets,
                    base_dim=base_dim,
                    data_cardinality=data_cardinality)
        self.base_net = []
        self.teacher_net = []
        self.fake_teacher_net = []
        self.teacher_inputs = []
        self.teacher_labels = []
        self.fake_teacher_inputs = []
        self.fake_teacher_labels = []
        self.which_teacher_dict = {} ## assigns students to teachers


        # make num_teachers copies of base nets, teachers, and fake teachers
        #store i-o of (fake) teachers
        for n in range(num_teachers):
            tn, ti, tl = self.add_nn(nn_dim=base_dim, make_ion=1) #base or teacher
            fn, fi, fl = self.add_nn(nn_dim=base_dim, make_ion=1) #base or teacher
            bn,_,_ = self.add_nn(nn_dim=base_dim, make_ion = 1)
            self.teacher_net.append(tn), self.teacher_inputs.append(ti)
            self.teacher_labels.append(tl)
            self.fake_teacher_net.append(fn)
            self.fake_teacher_inputs.append(fi)
            self.fake_teacher_labels.append(fl)
            self.base_net.append(bn)
        # the ith network has i/o-proj_mats which transport the base inputs
        # and outputs into the dimension of the ith network.
        self.input_proj_mat_list = list()
        self.output_proj_mat_list = list()


    def add_nn(self,nn_dim=None, make_ion=0):
        """
        Adds new nn. If not a teacher/base network, adds to list of networks: nn_index.
        network, adds base i/o data
        """
        new_nn = single_layer_net(nn_dim=nn_dim)
        if make_ion is 0:
            nn_family.add_nn(self,new_nn)
            self.set_which_teacher({len(self.nn_index)-1 : 0})
            input_proj_mat, output_proj_mat = self.create_projection_matrices(
                    big_dim=nn_dim)
            new_nn.all_synaptic_weights = output_proj_mat.transpose().dot(
               self.base_net[0].all_synaptic_weights).dot(input_proj_mat)

            self.input_proj_mat_list.append(input_proj_mat)
            self.output_proj_mat_list.append(output_proj_mat)
        if make_ion is 1:
            this_net = new_nn
            this_inputs = np.random.randn(nn_dim[0], self.data_cardinality)
            this_outputs = new_nn.forward_pass(this_inputs)
            return this_net, this_inputs, this_outputs

    def equalise_network_weights(self, nn_in, which_base_net=None):

        which_nn = self.nn_index.index(nn_in) #ie an integer
        wbn = which_base_net
        if wbn is None:
            wbn = self.which_teacher_dict[which_nn]
        input_proj_mat = self.input_proj_mat_list[which_nn]
        output_proj_mat = self.output_proj_mat_list[which_nn]

        nn_in.all_synaptic_weights = output_proj_mat.transpose().dot(
           self.base_net[wbn].all_synaptic_weights).dot(input_proj_mat)

    def get_n_vec(self, i_vec):
        return nn_family.get_n_vec(self, i_vec)

    def expected_steady_state_G(self, gamma=None, dt=None,
                                gamma_info=None, N_vec=None):
        return nn_family.expected_steady_state_G(self,gamma=gamma, dt=dt,
                                                 gamma_info=gamma_info,
                                                 N_vec=N_vec)

    def get_ith_io(self,i_val,is_online, real_teacher):
       # Gets inputs and outputs for ith network.
       wt = self.which_teacher_dict[i_val]
       ipm = self.input_proj_mat_list[i_val]
       opm = self.output_proj_mat_list[i_val]
       if real_teacher is 1:
           ti = self.teacher_inputs[wt]
           tl = self.teacher_labels[wt]
       elif real_teacher is 0:
           ti = self.fake_teacher_inputs[wt]
           tl = self.fake_teacher_labels[wt]

       if is_online is 1:
           print('Warning: cant do online training for the linear net')
       inputs_ = np.dot(ipm.transpose(), ti)
       outputs_ = np.dot(opm.transpose(),tl)
       return inputs_, outputs_

    def create_projection_matrices(self, big_dim=None, base_dim=None):
        if base_dim is None:
            base_dim = self.base_dim
        def make_orthonormal_mat(in_dimension, out_dimension):
            H = np.random.randn(in_dimension, in_dimension)
            Q, R = np.linalg.qr(H)  # Q is orthonormal matrix
            Q = np.delete(Q, [np.arange(in_dimension - out_dimension)], axis=0)
            return Q

        input_proj_mat = make_orthonormal_mat(big_dim[0], base_dim[0])
        output_proj_mat = make_orthonormal_mat(big_dim[1], base_dim[1])
        return input_proj_mat, output_proj_mat

    def fake_initialise(self, indices_to_run=None,
                         gamma_=None,
                         number_of_runs=1000,
                         delta_t=None,
                         gamma1il = [0.96, 0.04],
                         subsample_fraction=1):

        nn_family.fake_initialise(self, indices_to_run=indices_to_run,
                             gamma_=gamma_,
                             number_of_runs=number_of_runs,
                             delta_t=delta_t,
                             gamma1il = gamma1il,
                             subsample_fraction=subsample_fraction)

    def train_with_initialise(self, indices_to_run=None,
                     gamma_=None,
                     number_of_runs=None,
                     delta_t=None,
                     gamma1il = [0.96, 0.04], outs_class_in=None,
                     subsample_fraction=1,
                     is_online=0,
                     fake_runs=1000):
         """
         Does a fake initialise and a training run with the same input parameters
         """
         nn_family.train_with_initialise(self, indices_to_run=indices_to_run,
                          gamma_=gamma_,
                          number_of_runs=number_of_runs,
                          delta_t=delta_t,
                          gamma1il = gamma1il,
                          outs_class_in=outs_class_in,
                          subsample_fraction=subsample_fraction,
                          is_online=is_online,
                          fake_runs=fake_runs)


    def training_run(self, indices_to_run=[0,1],
                     gamma_=[0.5, 0.5, 0.1],
                     number_of_runs = 25,
                     delta_t = 1,
                     gamma1il=[0.96,0.04],
                     outs_class_in=None,
                     subsample_fraction=1,is_online=0, real_teacher=1):
        out_class = nn_family.training_run(self, indices_to_run=indices_to_run,
                               gamma_=gamma_,
                               number_of_runs=number_of_runs,
                               gamma1il=gamma1il,
                               delta_t=delta_t,
                               outs_class_in=outs_class_in,
                               subsample_fraction=subsample_fraction,
                               is_online=is_online, real_teacher=real_teacher)
        return out_class

class nn_fam_mlp:
    """
    want to add the same number of teachers, bases, and fake teachers respectively
    want (fake) teachers to have inputs and labels stored.
    """
    def __init__(self, num_nets=None,
             base_dim=None,
             data_cardinality=None,num_teachers=2):

        nn_family.__init__(self,num_nets=num_nets,
                    base_dim=base_dim,
                    data_cardinality=data_cardinality)
        self.base_net = []
        self.teacher_net = []
        self.fake_teacher_net = []
        self.teacher_inputs = []
        self.teacher_labels = []
        self.fake_teacher_inputs = []
        self.fake_teacher_labels = []
        self.which_teacher_dict = {} ## assigns students to teachers
        ## Make list of num_teachers different base nets.
        self.base_net,_,_ = zip(*[self.add_nn(nn_dim=base_dim,
                                  make_ion=1) for i in range(num_teachers)])
        self.base_net = list(self.base_net)

        self.base_num_layers = len(self.base_net[0].all_synaptic_weights)
        ########
        ## Make list of teacher nets, and store their i-o properties
        self.teacher_net,self.teacher_inputs,self.teacher_labels = \
        zip(*[self.add_nn(nn_dim=base_dim,
                                    make_ion=1) for i in range(num_teachers)])
        ## Make list of fake teachers (for fake initialise) and store i-o
        self.fake_teacher_net,self.fake_teacher_inputs,self.fake_teacher_labels = \
        zip(*[self.add_nn(nn_dim=base_dim,
                                    make_ion=1) for i in range(num_teachers)])

        ## Turn everything from tuples to lists: DOESN'T SEEM TO WORK OR MATTER
        for a in [self.teacher_inputs, self.teacher_labels,
                               self.fake_teacher_inputs, self.fake_teacher_labels,
                               self.teacher_net, self.fake_teacher_net]:
            a = list(a)


    def add_nn(self,nn_dim=None, make_ion=0, base_dims=0, equalise=True):
        """
        Adds new nn. If not a base network, adds to list of networks. If base
        network, adds base i/o data
        """
        nn_dim = np.array(nn_dim)
        if base_dims is 0:
            new_nn = s.neurNet(nn_dim)
        else:
            new_nn = s.neurNet(nn_dim[[0,1,-1]])

        if make_ion is 0:
            ## add to nn_index and assign to default teacher
            nn_family.add_nn(self,new_nn)
            self.set_which_teacher({len(self.nn_index)-1 : 0})
            #do stuff to equalise initial i-o properties with base net
            if equalise is True:
                # equalises to base_net 0. Need to manually equalise for others
                self.equalise_network_weights(new_nn)

        if make_ion is 1:
            this_net = new_nn
            this_inputs = np.random.randn(nn_dim[0], self.data_cardinality)
            this_outputs = new_nn.forward_pass(this_inputs)
            return this_net, this_inputs, this_outputs

    def set_which_teacher(self, input_dict):
        ## for ith nn_index, which teacher should i train on? key is nn_index
        #, value is which teacher
        nn_family.set_which_teacher(self,input_dict)

    def equalise_network_weights(self, nn_in, which_base_net=None):
        """
        # Make synaptic weights in new network such that output is
        equivalent to small network
        """
        wb = which_base_net
        if wb is None:
            wb = self.which_teacher_dict[self.nn_index.index(nn_in)]
        for i in np.arange(self.base_num_layers):
            nn_base_shape = self.base_net[wb].all_synaptic_weights[i].shape
            nn_in.layers[i].synaptic_weights.fill(0)
            #next line is so SW column that multiplies bias layer is still at the end
            nn_in.layers[i].synaptic_weights[:nn_base_shape[0],-1] \
            = self.base_net[wb].layers[i].synaptic_weights[:,-1]
            nn_in.layers[i].synaptic_weights[:nn_base_shape[0],:nn_base_shape[1]-1] \
            = self.base_net[wb].layers[i].synaptic_weights[:,:-1]

    def expected_steady_state_G(self, gamma=None, dt=None,
                                gamma_info=None, N_vec=None):
        return nn_family.expected_steady_state_G(self,gamma=gamma, dt=dt,
                                                 gamma_info=gamma_info,
                                                 N_vec=N_vec)

    def get_n_vec(self, i_vec):
        return nn_family.get_n_vec(self, i_vec)

    def get_ith_io(self,i_val,is_online,real_teacher):
        # i_val is only necessary in linear get_ith_io where it specifies
        # projection matrices. Redundant here

        wt = self.which_teacher_dict[i_val]
        rt = real_teacher
        if rt is 1:
            ti = self.teacher_inputs[wt]
            tl = self.teacher_labels[wt]
            tn = self.teacher_net[wt]
        elif rt is 0:
            ti = self.fake_teacher_inputs[wt]
            tl = self.fake_teacher_labels[wt]
            tn = self.fake_teacher_net[wt]

        if is_online is 0:
            return ti, tl
        elif is_online is 1:
            nn_dim=tn.layer_size_vec
            base_inputs = np.random.randn(nn_dim[0], self.data_cardinality)
            base_outputs = tn.forward_pass(base_inputs)
        return base_inputs, base_outputs

    def fake_initialise(self, indices_to_run=None,
                         gamma_=None,
                         number_of_runs=1000,
                         delta_t=None,
                         gamma1il = [0.96, 0.04],
                         subsample_fraction=1):

        nn_family.fake_initialise(self, indices_to_run=indices_to_run,
                             gamma_=gamma_,
                             number_of_runs=number_of_runs,
                             delta_t=delta_t,
                             gamma1il = gamma1il,
                             subsample_fraction=subsample_fraction)

    def train_with_initialise(self, indices_to_run=None,
                     gamma_=None,
                     number_of_runs=None,
                     delta_t=None,
                     gamma1il = [0.96, 0.04], outs_class_in=None,
                     subsample_fraction=1,
                     is_online=0,
                     fake_runs=1000):
         """
         Does a fake initialise and a training run with the same input parameters
         """
         nn_family.train_with_initialise(self, indices_to_run=indices_to_run,
                          gamma_=gamma_,
                          number_of_runs=number_of_runs,
                          delta_t=delta_t,
                          gamma1il = gamma1il,
                          outs_class_in=outs_class_in,
                          subsample_fraction=subsample_fraction,
                          is_online=is_online,
                          fake_runs=fake_runs)


    def training_run(self, indices_to_run=[0,1,2],
                     gamma_=[0.5, 0.5, 0.1],
                     number_of_runs = 25,
                     delta_t = 1,
                     subsample_fraction=1,
                     gamma1il=[0.96,0.04],
                     outs_class_in=None, is_online=0, real_teacher=1):
        out_class = nn_family.training_run(self, indices_to_run=indices_to_run,
                               gamma_=gamma_,
                               number_of_runs=number_of_runs,
                               gamma1il=gamma1il,
                               subsample_fraction=subsample_fraction,
                               delta_t=delta_t,
                               outs_class_in = outs_class_in,
                               is_online=is_online,
                               real_teacher=real_teacher)
        return out_class

class single_layer_net:
    #NB nn_dim = number of inputs, number of outputs.
    def __init__(self, nn_dim=[10, 5], input_weight_matrix = None, activation_function = 'identity'):
        if input_weight_matrix is not None:
            self.all_synaptic_weights = input_weight_matrix
        else:
            self.all_synaptic_weights = np.random.randn(nn_dim[1], nn_dim[0])
        self.set_activation_function(activation_function)
        self.layer_size_vec = nn_dim

    def set_activation_function(self, activation_function):
        if activation_function == 'identity':
            self.activation_function = lambda x : x
            self.activation_inverse_derivative = lambda x : np.zeros_like(x) + 1
        elif activation_function == 'sigmoid':
            self.activation_function = lambda x: 1/(1 + np.exp(-x))
            self.activation_inverse_derivative = lambda x: x*(1-x)

    def forward_pass(self, input_matrix):
        output_matrix = self.activation_function(np.dot(self.all_synaptic_weights, input_matrix))
        return output_matrix

    def get_grad_from_input(self, input_matrix, label_matrix):
         output_matrix = self.forward_pass(input_matrix)
         dEdo =  output_matrix - label_matrix
         dEdo *= self.activation_inverse_derivative(output_matrix)
         dEdw = np.dot(dEdo, input_matrix.transpose())
         return dEdw

    def make_noise(self):
        noise_ = np.random.randn(*self.all_synaptic_weights.shape)
        #noise_ -= dEdw_mat_in*dEdw_mat_in.flatten().dot(noise_.flatten())
        noise_ /= np.linalg.norm(noise_)
        return noise_

    def get_num_weights(self):
        return self.layer_size_vec[0]*self.layer_size_vec[1]

    def sum_weights(self, *argv):
        #sums weight-like objects (i.e. same dimensions as weights)
        return np.sum(argv, axis=0)

    def add_to_weights(self,direction):
        self.all_synaptic_weights += direction

    def scale_weights_in(self, weights_in, gamma_in):
        weights_in *= gamma_in


    def get_norm(self,weights_in):
        return np.linalg.norm(weights_in)

    def normalise_weights_in(self,weights_in,scalar_to_mult=1):
        """
        Assumes weights_in is mutable. Changes their value to normalise then
        multiplies by new_norm
        """
        norm_ = self.get_norm(weights_in)
        weights_in*=(scalar_to_mult/norm_)

    def get_error(self, input_matrix, label_matrix):
        output_matrix = self.forward_pass(input_matrix)
        return 0.5*np.sum( (output_matrix-label_matrix)**2)

    def get_g(self, input_matrix, label_matrix, direction=None,
              fd_step=0.0001, gradient_here=None,
              gradient_norm_here=None):
        #print('direction_shape=', direction.shape)
        #print('dir norm', np.linalg.norm(direction))
        delta_w = fd_step*direction / np.linalg.norm(direction)
        self.all_synaptic_weights += delta_w
        grad_f_w_plus_delta_w = self.get_grad_from_input(input_matrix,
                                                         label_matrix)
        self.all_synaptic_weights -= delta_w
        grad_f_w = gradient_here
        #print('grad here norm', np.linalg.norm(gradient_here))
        Hessian_times_delta_w = (grad_f_w_plus_delta_w - grad_f_w)/fd_step
        return (1/(2*gradient_norm_here))*(
                1/fd_step)*delta_w.flatten().dot(
                        Hessian_times_delta_w.flatten())



class output_class:
    """
    Stores info from training run. Needs to have: numbers of weights. gamma
    and gamma1il info.
    """
    def __init__(self, nn_family_=None):
        if nn_family_ is not None:
            self.sizes_vec = [nn_family_.nn_index[i].layer_size_vec  for i in range(
                                               len(nn_family_.nn_index))]

    def predict_k_from_g(self):
        gamma = self.gamma_info['gamma']
        dt_to_use = self.gamma_info['delta_t']
        norm_wdot_2_2 = (gamma[0]**2 + gamma[1]**2 + \
                             (self.n_vec/dt_to_use)*(gamma[2]**2))
        k = -(self.grad_norm_list/self.error_list)*(
                -gamma[0] + self.g_list*(self.step_norm**2)/(dt_to_use))
        #k = (self.grad_norm_list/self.error_list)*(
        #        -gamma[0] + self.g_list*dt_to_use*norm_wdot_2_2)
        self.predicted_ks = k

    def calc_k_from_run(self):
        dt_to_use = self.gamma_info['delta_t']
        err_vals=self.error_list
        k = -(err_vals[1:][:]-err_vals[:-1][:])/(dt_to_use*err_vals[:-1][:])
        k = np.vstack([k, np.zeros([1, k.shape[1]])])
        self.calced_ks = k

    def get_steady_states(self,which_thing,depth):
        #e.g outs_.get_steady_states(outs_.error_list,50)
        return np.mean(which_thing[-depth:][:],axis=0)


    def sort_v2_by_v1(v1,v2):
        #sorts elements of v2 according to the size of v
        sorted_pairs = [(v1[i],v2[i]) for i in v1.argsort()]
        return zip(*sorted_pairs)

    def get_nstar(self,C=None,gamma = None, dt=None):
        #only valid for linear case calculations
        if gamma is None:
            gamma = self.gamma_info['gamma']
        if dt is None:
            dt = self.gamma_info['delta_t']
        Nstar = []
        if C is None:
            print(gamma)
            for CC in np.asarray([0,1]): #These are theoretical bounds on C
                #assuming trace is psd. otherwise should be asarray[-1,1]
                Nstar.append((gamma[1]/gamma[2])*np.sqrt(
                    (1 + (gamma[1]/gamma[0])**2 )/(
                            CC*dt + ((gamma[2]/gamma[0])**2 ))))
        else:
            Nstar.append((gamma[1]/gamma[2])*np.sqrt(
                    (1 + (gamma[1]/gamma[0])**2 )/(
                            C*dt + (gamma[2]/gamma[0])**2 )))
        return Nstar

    def find_exp_nstar(self, gamma=None, dt = None, is_linear=1,Norig=None):
        if gamma is None:
            gamma = self.gamma_info['gamma']
        if dt is None:
            dt = self.gamma_info['delta_t']
        if is_linear is 1:
            pRoots = [1,
                      (dt*(gamma[0]/gamma[2])**2),
                      -((gamma[1]**2/gamma[2]**4)*(1 + gamma[0]**2)) ]
            if any(np.isnan(pRoots)) or any(np.isinf(pRoots)):
                return [0,0]
            else:
                return np.roots(pRoots)
        elif is_linear is 0:
            pRoots = [1, 0,
                      -( ((dt**2)*(gamma[0]**2 + gamma[1]**2)*(gamma[1]**2)/(gamma[2]**4))  ),
                      - ( ((dt*gamma[0])**2)/(gamma[2]**4))*Norig*(gamma[0]**2 + gamma[1]**2) ]
            if any(np.isnan(pRoots)) or any(np.isinf(pRoots)):
                return [0,0]
            else:
                return np.roots(pRoots)
            
    def test_exp_nstar(self,gamma=None, dt=None, Norig=None):
        #testing C = N^3/N*^2.  results in quartic with order 4 2 0 terms. so 
        #can solve as a quadratic in N^*^2.
        pRoots = [1,
                  -((dt*gamma[1]/gamma[2])**2)*(gamma[0]**2 + gamma[1]**2),
                   ((dt*gamma[1]/gamma[2])**2)*(gamma[0]**2 + gamma[1]**2)*Norig**3]
        nstarsquared = np.roots(pRoots)
        nstar = np.sqrt(nstarsquared)
        return nstar, nstarsquared


    def average_over_unique_vals(self,v1,v2):
        """
        v1 has repeated elements. returns unique(v1) and also
        average of values of v2 over each unique value of v1.
        So v1 = [1,2,2,4] and v2 = [3,4,6,5] returns:
            [1,2,4] and [3,5,5]
        """
        # Sort v1 index
        v1_index = np.argsort(v1)
        v1_sorted =  v1[v1_index]
        #process into unique elements and counts
        uniq, uniq_index, uniq_inv, uniq_count = np.unique(v1,
                                                           return_index=True,
                                                           return_inverse=True,
                                                           return_counts=True)
        #uniq_inv is now a code: identical elements within it correspond to
        # elements of v1 with the same values.

        uniq_v1 = v1[uniq_index]
        unique_mean = np.bincount(uniq_inv, weights=v2) / uniq_count
        uniq_second_moment = np.bincount(uniq_inv, weights=v2**2) / uniq_count
        uniq_var = (uniq_second_moment - unique_mean**2)
        uniq_std = np.sqrt(uniq_var)
        #bincount gives number of elements with same bincount identifier*weights
        return uniq_v1, unique_mean, uniq_std

    def __str__ (self):
        s1 = 'gamma_info: ' + str(self.gamma_info) + '\n'
        s2 = 'number_of_total_runs: ' + str(len(self.n_vec)) + '\n'
        s3 = 'number_of_unique_runs: ' + str(len(np.unique(self.n_vec))) + '\n'
        s4 = 'sizes of each run: ' + str(self.sizes_vec)
        return s1 + s2 + s3 + s4

    def merge_with_me(self,another_outs):
        """
        Concatenate two outs-classes. ONLY WORKS if gamma_infos and
        gamma1ils are identical.
        """
        for k in vars(self).keys():
            dot_k = getattr(self,k)
            if type(dot_k) is list:
                dot_k += getattr(another_outs,k)
            elif type(dot_k) is np.ndarray:
                dot_k = np.concatenate((dot_k,
                                        getattr(another_outs,k)),axis=-1)



if __name__ == '__main__':
    linear_on = 0
    #print('hi')
    if linear_on is 1:
        n_check = nn_fam_linear(base_dim=[10,5],
                                data_cardinality=1000)
        n_check.add_nn(nn_dim=[15,10])
        n_check.add_nn(nn_dim=[20,15])
        n_check.add_nn(nn_dim=[100,60])
        outs = output_class(nn_family_=n_check)
        n_check.training_run(indices_to_run=[0,1,2],
                                          gamma_=[0.1, 0.7, 0.],
                                          number_of_runs=250,
                                          delta_t=5,
                                          outs_class_in=outs)
        plt.plot(outs.error_list)
        plt.show()
    if linear_on is 0:
        n_check = nn_fam_mlp(base_dim=[10,5,5,5,5,5,10],
                             data_cardinality=1000)
        n_check.add_nn(nn_dim=[10,5,5,5,5,5,10])
        n_check.add_nn(nn_dim=[10,10,10,10,10,10,10])
        outs_ = output_class(nn_family_=n_check)
        n_check.train_with_initialise(indices_to_run=[0,1],
                                          gamma_=[0.005, 1, 0.],
                                          gamma1il = [0.96,0.04],
                                          number_of_runs = 3000,
                                          delta_t = 1, outs_class_in=outs_, is_online=0)
        n_check.expected_steady_state_G(gamma_info=outs_.gamma_info, N_vec=outs_.n_vec)
        k = outs_.calc_k_from_run()
        #plt.plot(outs_.g_list)
        plt.show()
        plt.plot(outs_.error_list)
        plt.legend()
        plt.show()
        plt.plot()
