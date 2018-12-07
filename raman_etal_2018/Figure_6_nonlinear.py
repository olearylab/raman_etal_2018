#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  5 15:39:22 2018
Objective of this script:
    1) Produce curve of steady state error (i.e. performance) against N
    for particular values of gamma and delta t
        a) in nonlinear case

@author: dvr23
"""

import numpy as np
import copy as copy
import setupMLP as s
import matplotlib.pyplot as plt
import matplotlib as mpl
#from multiprocessing import Pool
#from multiprocessing import Process, Queue
import pieces_new_resub as par
import os as os
import datetime as datetime
from matplotlib import gridspec
from scipy.interpolate import interp1d



def run_stack(sizes_to_test=None, gamma_to_use=None,dt_to_use=None,
              base_dim = [10,5], is_linear = 1):
    num_trials = len(sizes_to_test)
    if is_linear is 1:
        n_holder = par.nn_fam_linear(base_dim=base_dim,
                                     data_cardinality=1000)
    elif is_linear is 0:
        n_holder = par.nn_fam_mlp(base_dim=base_dim,
                                     data_cardinality=1000,
                                      num_teachers = num_repeats)


    for size_ in sizes_to_test:
        for i in range(num_repeats):
            n_holder.add_nn(nn_dim=size_)

    outs_all = [par.output_class() for i in [0,1]]
    v1list = []
    v2list = []
    v3list = []
    steady_errs = []
    def ith_run(i):
        n_holder.train_with_initialise(indices_to_run=range(num_repeats*num_trials),
                              gamma_=gamma_to_use,
                              number_of_runs = num_runs,
                              delta_t = dt_to_use,
                              gamma1il = [0.9, 0.1],
                              outs_class_in=outs_all[i])

        steady_errs.append(outs_all[i].get_steady_states(outs_all[i].error_list,50))

        v1,v2,v3 = outs_all[i].average_over_unique_vals(
                outs_all[i].n_vec, steady_errs[i])
        v1list.append(v1), v2list.append(v2), v3list.append(v3)
    ith_run(0)
    #[ith_run(i) for i in [0,1]]
    max_err = np.max(outs_all[i].error_list for i in [0,1])
    # =============================================================================
    # Get bounds on N*
    # =============================================================================

    Norig = n_holder.base_net[0].get_num_weights()
    bounds = outs_all[0].get_nstar()
    print(is_linear)
    opt = outs_all[0].find_exp_nstar(Norig=Norig, is_linear = is_linear)
    if np.isnan(opt[1]) is True:
        opt = [0,0]

    return v1list,v2list,v3list,bounds,opt,outs_all

# =============================================================================
# Calculations are done, now make figure:
# =============================================================================

if __name__ == '__main__':

# =============================================================================
#     This block of code chooses a number of neural network sizes, and plots
#     the number of synaptic weights against the steady state error. All
#     neural networks have the same i/o data.
#
# =============================================================================
    num_repeats = 2
    num_runs = 2000
    base_dim_ch = [10,5,10]
    is_linear = 0

    sizes_to_test = [[10,5,10], [10,15,10],
                     [10,30,10], [10,45,10],
                     [10,60,10],
                     [10,90,10], [10,120,10], [10,150,10],[10,200,10], [10,250,10],[10,300,10],[10,400,10]]


    sizes_vec = []
    gamma_vec = []
    dt_vec = []
    v1list = []
    v2list = []
    v3list= []
    outs_all = []
    bounds = []
    opt = []

    for i in range(3):
        sizes_vec.append(sizes_to_test)
        dt_vec.append(1)
    gamma_vec.append(np.asarray([0.04, 1.5, 0.03]))
    gamma_vec.append(np.asarray([0.04, 1.5, 0.04]))
    gamma_vec.append(np.asarray([0.04, 1.5, 0.05]))

    for i in range(3):
      a,b,c,d,e,f = run_stack(sizes_to_test=sizes_vec[i], gamma_to_use=gamma_vec[i],
                  dt_to_use=dt_vec[i], is_linear=is_linear, base_dim = base_dim_ch)

      v1list.append(a),v2list.append(b),v3list.append(c/np.sqrt(num_repeats)),bounds.append(d),
      opt.append(e), outs_all.append(f)



    bar_colour = []
    bar_colour.append((1/255)*np.asarray([217,95,2]))
    bar_colour.append((1/255)*np.asarray([117,112,179]))
    bar_colour.append((1/255)*np.asarray([27,158,119]))
    legend_labels_i = []
    for i in range(3):
        legend_labels_i.append('$\gamma = ' + np.array2string(
            gamma_vec[i], separator=', ') + '$')



    plt.style.use('seaborn-white')
    mpl.rc('text', usetex=True)
    fig_out = plt.figure()
    plt.locator_params(nbins=2)

    axs = fig_out.add_subplot(111)
    axs.set_ylabel('error (mean square)', fontsize=20)
    axs.locator_params(axis='y', nbins=4)
    axs.set_xlabel('number of synapses', fontsize=20)
    axs.tick_params(axis='both', which='major', labelsize=15)


    interps = [] #list of interpolating lines
    for i in range(3):
        axs.errorbar(v1list[i], v2list[i], yerr=v3list[i],
                     fmt = 'o',
                     ecolor=bar_colour[i],
                     mfc=bar_colour[i],
                     mec=bar_colour[i], markersize=10, linewidth=2,
                     label = legend_labels_i[i])

        #axs.axvline(x=bounds[0],ymin=0,ymax=1,lw=1)
        #axs.axvline(x=bounds[1],ymin=0,ymax=1,lw=1)
        #axs.axvspan(*bounds, alpha=0.5, facecolor=bar_colour[i])
        axs.axvline(x=np.max(opt[i]), ymin=0, ymax=1, color=bar_colour[i],
                    linestyle = '--', linewidth = 5)

        ## add interpolating line
        interps.append(interp1d(v1list[i][0], v2list[i][0], kind='nearest'))
        axs.plot(v1list[i][0], interps[i](v1list[i][0]), '-',color=bar_colour[i],visible='on')
    fig_out.legend(fontsize=18)
    if is_linear is 0:
        fig_out.title('nonlinear network')
    elif is_linear is 1:
        fig_out.title('linear network')
    plt.tight_layout(pad=0.4, w_pad=0.4, h_pad=0.4)


    np.savetxt('nonlinear_x_coords.csv', v1list[0])
    for i in np.arange(3):
        np.savetxt('nonlinear_means_' + str(i+1) +  '.csv', v2list[i])
        np.savetxt('nonlinear_st_errs_' + str(i+1) + '.csv', v3list[i])
    fig_out.set_size_inches([8,6])
    fig_out.tight_layout()
    fig_out.savefig('fig6_nonlinear.pdf')
