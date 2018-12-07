#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created October 2018
Make Figure 1
@author: dvr23
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare online learning from a distribution of Gaussian inputs, for different
network sizes. Compare  gamma3 = 0 case to gamma3 > 0  case
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

## add num_repeats copies of neural network of each size.

    for size_ in sizes_to_test:
        for i in range(num_repeats):
            n_holder.add_nn(nn_dim=size_)
            which_nn = len(n_holder.nn_index) - 1
            n_holder.set_which_teacher({which_nn : i})

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
                              outs_class_in=outs_all[i],
                              is_online=1)


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
    num_repeats = 12
    num_runs = 5000

    base_dim_ch = [10,5,5,5,5,5,10]
    base_dim_ch = [10,5,5,5,5,5,10]


    sizes_to_test = [[10,5,5,5,5,5,10],[10,12,12,12,12,12,10],
                     [10,17,17,17,17,17,10],
                     [10,22,22,22,22,22,10],
                     [10,27,27,25,27,27,10],[10,32,32,32,32,32,10],
                     [10,37,37,37,37,37 ,10], [10,45,45,45,45,45,10]]


    # =============================================================================


    sizes_vec = []
    gamma_vec = []
    dt_vec = []
    v1list = []
    v2list = []
    v3list= []
    outs_all = []
    bounds = []
    opt = []

## Doing everything twice to compare no-noise case to noise case
    for i in range(2):
        sizes_vec.append(sizes_to_test)
        dt_vec.append(1)
    gamma_vec.append(np.asarray([0.007, 1, 0.]))
    gamma_vec.append(np.asarray([0.007, 1, 0.05]))

    for i in range(2):
      a,b,c,d,e,f = run_stack(sizes_to_test=sizes_vec[i], gamma_to_use=gamma_vec[i],
                  dt_to_use=dt_vec[i], is_linear=0, base_dim = base_dim_ch)

      v1list.append(a),v2list.append(b),v3list.append(c/np.sqrt(num_repeats)),bounds.append(d),
      opt.append(e), outs_all.append(f)



    bar_colour = []
    bar_colour.append((1/255)*np.asarray([217,95,2]))
    bar_colour.append((1/255)*np.asarray([117,112,179]))
    bar_colour.append((1/255)*np.asarray([27,158,119]))
    legend_labels_i = []
    for i in range(2):
        legend_labels_i.append('$\gamma = ' + np.array2string(
            gamma_vec[i], separator=', ') + '$')



    plt.style.use('seaborn-white')
    mpl.rc('text', usetex=True)
    fig_out = plt.figure()

    ax_nonoise = fig_out.add_subplot(222)
    ax_noise = fig_out.add_subplot(224,sharex = ax_nonoise)


    axlistSt = [ax_nonoise, ax_noise]
    for ax in axlistSt:
        ax.set_ylabel('steady-state error', fontsize=15)
        ax.locator_params(axis='y', nbins=4)
        ax.set_xlabel('network size (synapses)', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)




    for i in range(len(axlistSt)):
        axlistSt[i].errorbar(v1list[i], v2list[i], yerr=v3list[i],
                     fmt = 'o',
                     ecolor=bar_colour[i],
                     mfc=bar_colour[i],
                     mec=bar_colour[i], markersize=10, linewidth=2,
                     label = legend_labels_i[i])

        #axs.axvline(x=bounds[0],ymin=0,ymax=1,lw=1)
        #axs.axvline(x=bounds[1],ymin=0,ymax=1,lw=1)
        #axs.axvspan(*bounds, alpha=0.5, facecolor=bar_colour[i])
#        axs.axvline(x=np.max(opt[i]), ymin=0, ymax=1, color=bar_colour[i],
#                    linestyle = '--', linewidth = 5)
    for ax in axlistSt:
        ax.legend(fontsize=18)
    plt.tight_layout(pad=0.4, w_pad=0.4, h_pad=0.4)

    ax_nonoise_run = fig_out.add_subplot(221)
    ax_noise_run = fig_out.add_subplot(223, sharex = ax_nonoise_run)
    axRuns = [ax_nonoise_run, ax_noise_run]
    which_to_run1 = np.asarray([1,91,-29]) +2
    which_to_run2 = np.asarray([1,91,-29]) +2
    color_ = [tuple((1/255)*np.asarray([254,232,200])),
          tuple((1/255)*np.asarray([253,187,132])),
          tuple((1/255)*np.asarray([227,74,51]))]


    for i,w in enumerate(which_to_run1):
        ax_nonoise_run.plot(outs_all[0][0].error_list[:,w], label=
                            outs_all[0][0].n_vec[w],color=color_[i])

    for i,w in enumerate(which_to_run2):
        ax_noise_run.plot(outs_all[1][0].error_list[:,w], label=
                            outs_all[1][0].n_vec[w],color=color_[i])
    for ax in axRuns:
        ax.legend(fontsize=18)
        ax.set_ylabel('error (mean-square)', fontsize=15)
        ax.locator_params(axis='y', nbins=4)
        ax.set_xlabel('time (training cycles)', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
