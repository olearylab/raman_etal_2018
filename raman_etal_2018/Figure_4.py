#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 30 11:24:01 2018
Generate each figure panel necessary for the main paper as a function.
@author: dvr23

Functions:
    k_pred_plot implements the density plot in Figure 4

    outs, figs = gGrowthFig()
        Shows a run of linear (left) and nonlinear (right). Training curves
        are on top, and g curves are on bottom. Used in Figure 4

The other functions are not used in the paper, but explore different ways of
visualising the core concepts of the paper.


"""

import numpy as np
import scipy.io as sio
import pieces_new as par
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib as matplotlib
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d import Axes3D
import copy as copy

""" Colours are:
   27 158 119
   217 95 2
   117 112 179
"""
colour = []
colour.append((1/255)*np.asarray([217,95,2]))
colour.append((1/255)*np.asarray([117,112,179]))
colour.append((1/255)*np.asarray([27, 158, 119]))



def do_plot(outs_,fig1,gs,col_num, top_y_topin = None, bot_y_topin = None, num_lines = [0,1]):
    """
    Generic function for cleaning out 2x2 set of figures with training curves
    on top and local task difficulty curves on bottom.
    fig1 is already set as the current figure from the input.
    gs is the gridspec. (2,2) = 2x2 grid of subplots
    e.g. subplot(gs[0,:] makes the subplot span all columns, and first row)
    subplot(sharey=ax) shares yaxis with ax
    """

    num_runs = outs_.error_list.shape[0]

    ax0 = plt.subplot(gs[0,col_num], sharey = top_y_topin)

    line0 = [ax0.plot(outs_.error_list[:,i], color=colour[i]) for i in num_lines]
    ax0.set_ylabel(' MSE')
    #the second subplot
    # shared axis X
    ax1 = plt.subplot(gs[1,col_num], sharex = ax0, sharey = bot_y_topin)
    line1 = [ax1.plot(outs_.g_list[:,i], color=colour[i]) for i in num_lines]
    line_pred = ax1.plot([0, num_runs],
                         [outs_.expected_g_vec[0], outs_.expected_g_vec[0]],
                         color='k',
                         linestyle='--', linewidth=2)
    #plt.setp(ax0.get_xticklabels(), visible=False)
    # remove last tick label for the second subplot
    #yticks = ax1.yaxis.get_major_ticks()
    #yticks[-1].label1.set_visible(False)
    ax1.set_ylabel('Local Task \n Diff. ')
    ax1.set_xlabel('Learning Cycles')
    ax1.set_ylim(0)
    ax0.set_ylim(0)
    # put legend on first subplot
    #ax0.legend((line0, line1), ('red line', 'blue line'), loc='upper right')

    # remove vertical gap between subplots
    fig1.subplots_adjust(hspace=.1)
    #return fig1

def motivation_plot():
    plt.style.use('ggplot')
    gamma_to_use = [0.02, 1, 0]
    dt_to_use = 1
    num_runs = 1000
    n_mlp = par.nn_fam_mlp(base_dim=[10,5,5,6],
                                data_cardinality=1000)
    n_mlp.add_nn(nn_dim=[10,5,5,6])
    n_mlp.add_nn(nn_dim=[10,10,10,6])
    outs_ = par.output_class(nn_family_=n_mlp)
    n_mlp.training_run(indices_to_run=[0,1],
                                          gamma_=gamma_to_use,
                                          number_of_runs=num_runs,
                                          delta_t=dt_to_use,
                                          gamma1il=[0.9,0.1],
                                          outs_class_in=outs_)
    fig_in = plt.figure()
    ax = fig_in.add_subplot(111)
    ax.plot(outs_.error_list)
    l0 = ax.lines[0]
    l1 = ax.lines[1]
    l0.set_color([217/255,95/255,2/255  ])
    l1.set_color([117/255,112/255,179/255  ])
    ax.set_xlabel('Time (Learning Cycles)', fontsize = 16)
    ax.set_ylabel('MSE', fontsize = 16)
    ax.xaxis.set_ticks(np.arange(0, num_runs, num_runs/2))
    mm = np.round(np.max(outs_.error_list), decimals = -2)
    ax.yaxis.set_ticks(np.arange(0, 1200, 400))
    ax.xaxis.set_ticks(np.arange(0, 1500, 500))
    plt.setp(ax.get_xticklabels(), fontsize=16)
    plt.setp(ax.get_yticklabels(), fontsize=16)
    return fig_in, outs_

def motivation_plot_new():
    """
    Four panels:
        (1,1): bigger vs smaller network MSE over time, gamma3=0
        (1,2): Steady state error over time, gamma3 = 0
        (2,1) three sizes of network: too small, good, too big. learning curves
        over time
        (2,2): Steady state error for (2,1) values of gamma
    """
    plt.style.use('ggplot')
    figs = plt.figure()
    ax11 = figs.add_subplot(221)
    ax21 = figs.add_subplot(223)

    gamma_first = [0.006,1,0]
    gamma_second = [0.06,1,0.15]
    dt_to_use = 1
    num_runs=2000

    n_mlp = par.nn_fam_mlp(base_dim=[10,5,5,6],
                                data_cardinality=1000)
    n_mlp.add_nn(nn_dim=[10,5,5,6])
    n_mlp.add_nn(nn_dim=[10,10,10,6])
    n_mlp.add_nn(nn_dim=[10,40,40,6])
    outs_nonoise = par.output_class(nn_family_=n_mlp)
    outs_noise = par.output_class(nn_family_=n_mlp)
    n_mlp.training_run(indices_to_run=[0,1,2],
                                          gamma_=gamma_first,
                                          number_of_runs=num_runs,
                                          delta_t=dt_to_use,
                                          gamma1il=[0.9,0.1],
                                          outs_class_in=outs_nonoise)
    n_mlp.training_run(indices_to_run=[0,1,2],
                                          gamma_=gamma_second,
                                          number_of_runs=num_runs,
                                          delta_t=dt_to_use,
                                          gamma1il=[0.9,0.1],
                                          outs_class_in=outs_noise)

    ax11.plot(outs_nonoise.error_list)
    ax21.plot(outs_noise.error_list)
    return [outs_nonoise, outs_noise], figs



def gGrowthFig():
    """"
    First plot: linear network. Show error curve, and that G is predictable
    a priori
    """
    colors = [[217/255,95/255,2/255  ], [117/255,112/255,179/255  ]]
    plt.style.use('seaborn-white')
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])

    gamma_to_use = [0.2, 1, 0]
    dt_to_use = 2
    num_runs = 500
    n_linear = par.nn_fam_linear(base_dim=[10,6],
                                data_cardinality=1000)
    n_linear.add_nn(nn_dim=[15,8])
    n_linear.add_nn(nn_dim=[20,10])
    outs_lin = par.output_class(nn_family_=n_linear)
    n_linear.training_run(indices_to_run=[0],
                                          gamma_=gamma_to_use,
                                          number_of_runs=num_runs,
                                          delta_t=dt_to_use,
                                          gamma1il=[0.9,0.1],
                                          outs_class_in=outs_lin)
    #outs_lin.error_list /= np.max(outs_lin.error_list)

    #do_plot(outs_lin, fig_in,gs,0, num_lines = [0])


    n_mlp = par.nn_fam_mlp(base_dim=[10,8,6],
                                data_cardinality=1000)
    n_mlp.add_nn(nn_dim=[10,8,6])
    n_mlp.add_nn(nn_dim=[10,20,6])
    outs_nlin = par.output_class(nn_family_=n_mlp)
    n_mlp.training_run(indices_to_run=[0],
                                          gamma_=gamma_to_use,
                                          number_of_runs=num_runs,
                                          delta_t=dt_to_use,
                                          gamma1il=[1,0],
                                          outs_class_in=outs_nlin)
    #outs_nlin.error_list /= np.max(outs_nlin.error_list)
    #axs = fig_in.axes
    #do_plot(outs_nlin,fig_in,gs,1,top_y_topin = axs[0],
    #        bot_y_topin = axs[1], num_lines = [0])

    outs_ = [outs_lin, outs_nlin]
    #axs = fig_in.axes
    #four_panel_process(fig_in,axs)
    gs = gridspec.GridSpec(ncols=3, nrows = 3, height_ratios=[1,0.2,0.5],
                           width_ratios = [2,0.5,2])
    fig_in = plt.figure()
    ax00 = fig_in.add_subplot(gs[0,0])
    ax01 = fig_in.add_subplot(gs[0,2])
    ax10 = fig_in.add_subplot(gs[2,0], sharex=ax00)
    ax11 = fig_in.add_subplot(gs[2,2],sharex=ax01,sharey=ax10)
    ax10.set_ylabel('local task \n difficulty')
    ax00.set_ylabel('error \n (mean square)')
    ax10.set_xlabel('time \n (learning cycles)')
    ax11.set_xlabel('time \n  (learning cycles)')
    ax00.plot(outs_lin.error_list,color=colors[0])
    ax01.plot(outs_nlin.error_list,color=colors[0])
    ax10.plot(outs_lin.g_list,color=colors[0])
    ax11.plot(outs_nlin.g_list,color=colors[0])
    ax00.set_yticks([0,20000,40000,60000])
    ax01.set_yticks([0,200,400,600])
    for a in fig_in.axes:
        a.set_xticks([0,250,500])

    for a in [ax10,ax11]:
        a.set_yticks([0,0.1,0.2])
        a.set_ylim(0)
        a.axhline(y=outs_lin.expected_g_vec[0],xmin=0,xmax=1,
                  lw=2, ls='--',color='k')
    return outs_, fig_in


def four_panel_process(fig_in, axs):
    """
    Gets rid of xlabels/ticks on top panels, y labels/ticks on right panels,
    formats stuff
    """
    [axs[i]. set_ylabel('') for i in [2,3]]
    [plt.setp(axs[i].get_yticklabels(), visible=False) for i in [2,3]]
    [plt.setp(axs[i].get_xticklabels(), visible=False) for i in [0,2]]
    top_left_labels = axs[0].get_yticklabels()
    top_right_labels = axs[1].get_yticklabels()
    [t.set_visible(False) for t in top_left_labels]
    [top_left_labels[i].set_visible(True) for i in [0,-2]]
    [t.set_visible(False) for t in top_right_labels]
    [top_right_labels[i].set_visible(True) for i in [0,-2]]

    bl_labels = axs[1].get_xticklabels()
    br_labels = axs[3].get_xticklabels()
    [t.set_visible(False) for t in bl_labels]
    [t.set_visible(False) for t in br_labels]
    [bl_labels[i].set_visible(True) for i in [1,-2]]
    [br_labels[i].set_visible(True) for i in [1,-2]]
    fig_in.set_tight_layout(True)
    # for i in range(4):
    #    plt.setp(axs[i].get_xticklabels(), fontsize=18)

def small_big_compare_training_curves():
    """"
    Nonlinear network. Small and big N. Compare training curves and local
    task difficultiess on left panel. Compare same on right panel, but with
    subsampled gradient.
    """

    plt.style.use('ggplot')
    colors = [[217/255,95/255,2/255  ], [117/255,112/255,179/255  ]]
    matplotlib.rcParams.update({'font.size': 12})
    gs = gridspec.GridSpec(ncols=4, nrows = 3, height_ratios=[2,0.5,2],
                           width_ratios = [4,1,4,1])
    fig_in = plt.figure()
    gamma_to_use = [0.04, 1, 0]
    dt_to_use = 1
    num_runs = 2000
    n_mlp = par.nn_fam_mlp(base_dim=[10,3,10],
                                data_cardinality=1000)
    n_mlp.add_nn(nn_dim=[10,5,10])
    n_mlp.add_nn(nn_dim=[10,100,10])
    nn_big =n_mlp.nn_index[1]
    #for i in range(len(nn_big.layers)):
    #    if i is 0:
    #        nn_big.layers[i].re_initialise()

    outs_ = par.output_class(nn_family_=n_mlp)
    n_mlp.training_run(indices_to_run=[0,1],
                                          gamma_=gamma_to_use,
                                          number_of_runs=num_runs,
                                          delta_t=dt_to_use,
                                          gamma1il=[0.9,0.1],
                                          outs_class_in=outs_)

    #fig_out1 =
    #do_plot(outs_, fig_in, gs,0)
    fig_in = plt.figure()
    ax00 = fig_in.add_subplot(gs[0,0])
    ax01 = fig_in.add_subplot(gs[0,2])
    ax02 = fig_in.add_subplot(gs[0,3],sharey=ax01)
    ax10 = fig_in.add_subplot(gs[2,0],sharex=ax00,sharey=ax00)
    ax11 = fig_in.add_subplot(gs[2,2],sharex=ax01,sharey=ax01)
    ax12 = fig_in.add_subplot(gs[2,3],sharex=ax02, sharey=ax11)
    ax01.plot(outs_.error_list)
    ax00.plot(outs_.g_list)


    #figs.append(fig_out1)
    outs_subsample = par.output_class(nn_family_=n_mlp)
    n_mlp.training_run(indices_to_run=[0,1],
                                          gamma_=gamma_to_use,
                                          number_of_runs=num_runs,
                                          delta_t=dt_to_use,
                                          gamma1il=[0.9,0.1],
                                          outs_class_in=outs_subsample,
                                          subsample_fraction=0.001)

    outs_all = [outs_,outs_subsample]
    ax11.plot(outs_subsample.error_list)
    ax10.plot(outs_subsample.g_list)



    ax00.axhline(y=outs_.expected_g_vec[0],xmin=0,xmax=1,
                 lw=2, ls='--',color='k')
    ax00.set_ylabel('Local task \n difficulty')
    ax10.set_ylabel('Local task \n difficulty')
    ax01.set_ylabel('MSE')
    ax11.set_ylabel('MSE')
    ax10.set_xlabel('Learning cycles')
    ax11.set_xlabel('Learning cycles')
    ax00.set_title('No subsampling',pad=10)
    ax10.set_title('Subsampling',pad=10)


    #Now run a few times and end up with mean values for steady state error
    num_reps = 0
    means = np.zeros([2,num_reps])
    means_subsample = np.zeros([2,num_reps])
    for n in range(num_reps):
        outs_rep = par.output_class(nn_family_=n_mlp)
        outs_repsub = par.output_class(nn_family_=n_mlp)
        n_mlp.training_run(indices_to_run=[0,1],
                                          gamma_=gamma_to_use,
                                          number_of_runs=num_runs,
                                          delta_t=dt_to_use,
                                          gamma1il=[0.9,0.1],
                                          outs_class_in=outs_rep)
        means[:,n] = np.mean(outs_rep.error_list[-100:,:],axis=0).transpose()


        n_mlp.training_run(indices_to_run=[0,1],
                                          gamma_=gamma_to_use,
                                          number_of_runs=num_runs,
                                          delta_t=dt_to_use,
                                          gamma1il=[0.9,0.1],
                                          outs_class_in=outs_repsub,
                                          subsample_fraction=0.001)

        means_subsample[:,n] = np.mean(outs_repsub.error_list[-100:,:],
                       axis=0).transpose()

    gen_error = np.mean(means,axis=1)
    gen_std = np.std(means,axis=1)
    sub_error = np.mean(means_subsample,axis=1)
    sub_std = np.mean(means_subsample,axis=1)
    print(means)
    print(means_subsample)
    print(gen_error)
    print(sub_error)
    ax02.errorbar(1,gen_error[0],yerr=gen_std[0],fmt='o',color=colors[0])
    ax02.errorbar(1,gen_error[1],yerr=gen_std[1],fmt='o',color= colors[1])
    ax12.errorbar(1,sub_error[0],yerr=sub_std[0],fmt='o',color=colors[0])
    ax12.errorbar (1,sub_error[1],yerr=sub_std[1],fmt='o',color= colors[1])

    first_axes = [ax00,ax01,ax02,ax10]
    for a in fig_in.axes:
        if (len(a.lines)>1) is True:
            a.lines[0].set_color(colors[0])
            a.lines[1].set_color(colors[1])
    ax01.lines[0].set_label('small')
    ax01.lines[1].set_label('big')
    fig_in.legend(fontsize=15)
    [plt.setp(a.get_yticklabels(), visible=False) for a in [ax02,ax12]]
    [plt.setp(a.get_yticklines(), visible=False) for a in [ax02,ax12]]
    [plt.setp(a.get_xticklines(), visible=False) for a in [ax02,ax12]]
    [plt.setp(a.get_xticklabels(), visible=False) for a in [ax00,ax01,ax02,ax12]]
    return outs_all, fig_in,means, means_subsample

def small_big_compare_training_curves_linear():
    """
    Linear network. Small and big N. Compare training curves and local task
    difficulties
    """


    return outs_all, fig_in

def k_pred_plot():
    """
    Here we want to show that our expected value of k corresponds to that
    experienced by the model. Do linear and nonlinear, small and
    large error, to get 2x2 plots

    """
    dt_to_use = 2
    num_runs = 1000
    n_lin = par.nn_fam_linear(base_dim=[10,5],
                                data_cardinality=1000)
    n_lin.add_nn(nn_dim=[10,5])
    n_nlin = par.nn_fam_mlp(base_dim=[10,5,5,10],
                            data_cardinality=1000)
    n_nlin.add_nn(nn_dim=[10,5,5,10])
    outs_all = [[par.output_class() for i in [0,1]] for j in [0,1] ]

    def get_ij_data(i,j):
        if j is 0:
            n_x = n_lin
        elif j is 1:
            n_x = n_nlin
        if i is 0:
            gamma_to_use = [1, 0.05 ,0.05]
        elif i is 1:
            gamma_to_use = [0.2, 0.5 ,0.1]
        n_x.training_run(indices_to_run=[0],
                                              gamma_=gamma_to_use,
                                              number_of_runs=num_runs,
                                              delta_t=dt_to_use,
                                              gamma1il=[1,0],
                                              outs_class_in=outs_all[i][j])

        outs_all[i][j].predict_k_from_g()
        outs_all[i][j].calc_k_from_run()


    [get_ij_data(i,j) for i in [0,1] for j in [0,1]]
    fig_in = plt.figure()


    # Calculate the point density

    def draw_subplot(i,j,figg,gss):
        xy = np.vstack([np.squeeze(outs_all[i][j].calced_ks),
                        np.squeeze(outs_all[i][j].predicted_ks)])
        z = gaussian_kde(xy)(xy)

        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = np.squeeze(outs_all[i][j].calced_ks)[idx], \
        np.squeeze(outs_all[i][j].predicted_ks)[idx], z[idx]

        ax = plt.subplot(gss[i,j])

        cax = ax.scatter(x, y, c=z, s=50, edgecolor='')
        fig_in.colorbar(cax)
        #plt.scatter(outs_.calced_ks, outs_.predicted_ks)
        min_pt = np.min([outs_all[i][j].predicted_ks,
                         outs_all[i][j].calced_ks])
        max_pt = np.max([outs_all[i][j].predicted_ks,
                         outs_all[i][j].calced_ks])
        plt.plot([min_pt,max_pt],[min_pt,max_pt],
                 color='black',linestyle='--', linewidth=3)
        if i is 1:
            ax.set_xlabel('k')
        if j is 0:
            ax.set_ylabel('$k_{pred}$')

    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], wspace = 0.5)
    plt.style.use('ggplot')
    matplotlib.rcParams.update({'font.size': 10})
    plt.locator_params(nbins=2)
    [draw_subplot(i,j,fig_in,gs) for i in[0,1] for j in [0,1]]
    fig_in.set_tight_layout(True)
    return outs_all, fig_in


def time_to_MSE():
    """
    This function plots MSE on y axis, and first passage time to that level of
    MSE on x axis. First passage time relative to perfect gradient descent?
    Want 4 levels of systematic error, all on same neural
    network.
    """
    dt_to_use = 1
    num_runs = 1000
    num_repeats = 4
    num_tests = 4
    n_mlp = par.nn_fam_mlp(base_dim=[10,5,5,10],
                                data_cardinality=1000)
    n_mlp_big = copy.deepcopy(n_mlp)
    plt.style.use('ggplot')
    matplotlib.rcParams.update({'font.size': 16})
    matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
    ## for Palatino and other serif fonts use:
    #rc('font',**{'family':'serif','serif':['Palatino']})
    matplotlib.rc('text', usetex=True)
    figs = plt.figure(figsize=(10, 10))
    ax1 = figs.add_subplot(1,2,1)
    ax2 = figs.add_subplot(1,2,2, sharey = ax1, sharex = ax1)
    ax = figs.axes
    for j in range(num_tests*num_repeats):
        n_mlp.add_nn(nn_dim=[10,10,10,10])
        n_mlp_big.add_nn(nn_dim=[10,60,60,10])

    def jth_plot(j,which_ = None,outs_in=None):
        if which_ is 1:
            mlp_to_use = n_mlp
        elif which_ is 2:
            mlp_to_use = n_mlp_big
        print(j)
        mlp_to_use.training_run(indices_to_run=range(j,j+num_repeats),
                                  gamma_=gamma_to_use[j],
                                  number_of_runs=num_runs,
                                  delta_t=dt_to_use,
                                  gamma1il=[0.98,0.02],
                                  outs_class_in=outs_in[j],
                                  subsample_fraction=1)

    def tmse_parse(error_list, dt_used, err_vals):
        #given error list of shape [time_vec, num_runs], get out vector of
        #time to get to each error
        # err_vals = np.arange(error_list.min(),error_list.max())
        time_to_vals = np.zeros([np.max(err_vals.shape),num_repeats])
        #print(max(err_vals.shape), min(error_list.shape))
        for e in np.arange(max(err_vals.shape)):
            rr,cc = np.where(error_list < err_vals[e]) #rows and cols
            gott = np.zeros(num_repeats)
            for f in np.arange(num_repeats):
                if len(rr[cc==f]) is not 0:
                   gott[f] = rr[cc==f].min()
                if len(gott[gott==0]) is 0:
                    time_to_vals[e] = np.mean(gott[gott.nonzero()])
                else:
                    time_to_vals[e] = np.nan
            time_to_vals *= dt_used
        return np.mean(time_to_vals,axis=1)

    gamma_to_use = []
    outs_all1 = []
    outs_all2 = []
    time_to_vals1 = []
    time_to_vals2 = []
    color_ = [tuple((1/255)*np.asarray([254,240,217])),
              tuple((1/255)*np.asarray([253,204,138])),
              tuple((1/255)*np.asarray([252,141,89])),
              tuple((1/255)*np.asarray([215,48,31]))]

    def run_stuff(j,which_=None):
        if which_ is 1:
            outs_all = outs_all1
            time_to_vals = time_to_vals1
            gamma_to_use.append([0.05,0.5*j,0])
        else:
            outs_all = outs_all2
            time_to_vals = time_to_vals2
        outs_all.append(par.output_class())
        #gamma_to_use.append([0.05,0,0.05*j])
        jth_plot(j,which_=which_, outs_in = outs_all)

        err_vals = np.arange(outs_all[j].error_list.min(),
                             outs_all[j].error_list.max())
        time_to_vals.append(tmse_parse(outs_all[j].error_list,dt_to_use,err_vals))
        str_  ="$\gamma = $" + str(gamma_to_use[j])
        ax[which_-1].plot(time_to_vals[j],err_vals, label = str_,
          color = color_[j], linewidth=4)
        ax[which_-1].set_ylabel('MSE')
        ax[which_-1].set_xlabel('Time to reach')
        ax[which_-1].set_yscale('log')
        ax[which_-1].legend()
    for j in range(num_tests):
        for i in range(2):
            run_stuff(j,which_=i+1)



        #ax2.plot(time_to_vals[j]/time_to_vals[0], err_vals)
    return figs, [outs_all1, outs_all2], [time_to_vals1, time_to_vals2]




def lr_vs_err(attr_str_x = 'error_list', attr_str_y = 'calced_ks'):
    """
    Want to plot learning rate vs absolute error. Show that learning rates are
    similar when absolute error high, and that discrepancy only arises later on
    In fact this function plots any properties of outs_list against each other
    """
    dt_to_use = 1
    num_runs = 300
    num_repeats = 4
    num_tests = 4
    n_mlp = par.nn_fam_mlp(base_dim=[10,5,5,6],
                                data_cardinality=1000)
    gamma_to_use = []
    outs_all = []
    for i in range(num_tests*num_repeats):
        n_mlp.add_nn(nn_dim=[10,20,20,6])
    #gamma_to_use .append([0.02,0.8,0.])
    plt.style.use('ggplot')
    matplotlib.rcParams.update({'font.size': 12})
    gs = gridspec.GridSpec(num_tests, 2, height_ratios=[2]*num_tests,
                           width_ratios = [1,1], hspace=0.4)
    figs = plt.figure(figsize=(4, 8))
    axL = [figs.add_subplot(gs[j,:]) for j in range(num_tests)]
    [axL[j].set_xticklabels('') for j in range(num_tests-1)]
    [axL[j].get_shared_x_axes().join(axL[j], axL[j-1]) for j in range(1,num_tests)]
    [axL[j].get_shared_y_axes().join(axL[j], axL[j-1]) for j in range(1,num_tests)]
    err_vecs = [[] for j in range(num_tests)]
    k_vecs = copy.deepcopy(err_vecs)
    outs_all = [par.output_class() for j in range(num_tests)]
    def runningMeanFast(x, N):
        return np.convolve(x, np.ones((N,))/N)[(N-1):]

    def jth_run(j):
        #jth test, i repeats
        gamma_to_use.append([0.05, 0.8*(j),0.*j])
        n_mlp.training_run(indices_to_run=range(j,j+num_repeats) ,
                              gamma_=gamma_to_use[j],
                              number_of_runs=num_runs,
                              delta_t=dt_to_use,
                              gamma1il=[0.98,0.02],
                              outs_class_in=outs_all[j],
                              subsample_fraction=1)
        outs_all[j].calc_k_from_run()
        err_vecs[j] = np.reshape(getattr(outs_all[j],attr_str_x),(-1))
        err_args = err_vecs[j].argsort()
        err_vecs[j] = err_vecs[j][err_args]
        k_vecs[j] = np.reshape(getattr(outs_all[j],attr_str_y),(-1))
        k_vecs[j] = k_vecs[j][err_args]
    [jth_run(j) for j in range(num_tests)]
    for j in range(num_tests):
        #axL[j].scatter(err_vecs[j],k_vecs[j], edgecolor='', s=3)
        axL[j].plot(err_vecs[0],k_vecs[0],color='k', linestyle = '--')
        if j > 0:
            axL[j].plot(err_vecs[j],runningMeanFast(k_vecs[j],num_repeats*10),color='b',linewidth=2, alpha = 0.5)
    return outs_all, figs, err_vecs, k_vecs

def opt_N_curve():
    """
    Draw N on y axis, gamma3 on x axis. gamma1 is constant. Two different values of gamma 2

    """
    outs = par.output_class()
    def get_N(gamma3, is_linear):
        ns = outs.find_exp_nstar(gamma=[gamma1,gamma2,gamma3], dt = 1,
                            is_linear=is_linear,Norig=200)
        return np.max(ns)

    gamma1 = 0.1
    gamma2 = 1
    gamma3vec = np.linspace(0.001,0.05,1000)
    Nvec = [get_N(x,1) for x in gamma3vec]
    matplotlib.rc('text', usetex=True)
    figs = plt.figure(figsize=(8,3))

    ax1 = figs.add_subplot(141)
    ax1.set_title('Linear Network')
    ax1.plot(gamma3vec,Nvec, label = 'Bad learning rule')
    gamma2 = 0.01
    Nvec = [get_N(x,1) for x in gamma3vec]
    ax1.plot(gamma3vec,Nvec)
    ax1.plot(gamma3vec,Nvec, label = 'Good learning rule')



    ax2 = figs.add_subplot(143,sharey=ax1)
    ax2.set_title('Nonlinear network')
    gamma2 = 1
    Nvec = [get_N(x,0) for x in gamma3vec]
    ax2.plot(gamma3vec,Nvec)
    gamma2 = 0.01
    Nvec = [get_N(x,0) for x in gamma3vec]
    ax2.plot(gamma3vec,Nvec)
    ax2.plot(gamma3vec,Nvec)



    [ax.set_yscale('log') for ax in [ax1,ax2]]
    [ax.set_ylabel('$N^*$',fontsize=16) for ax in [ax1,ax2]]
    [ax.set_xlabel('$\gamma_3$',fontsize=16) for ax in [ax1,ax2]]
    figs.legend()

    return outs, figs



def randomness_lr_plot():
    """
    Here we want to show (in e.g. an MLP) that learning rate is fast when task
    difficulty is low, even for very approximate learning rules. To do so, we
    need plot of learning rate against local task difficulty, directly above
    plot of training errors.

    Show run with very large error, and run with very small error. Do histogram
    of local task difficulty against learning rate. These should have identical
    distributions but different sampling densities: the high error will always
    be at a high task difficulty.

    Then show small vs large networks learning with same gamma,T. At low task
    difficulties, the densities should intersect. At higher task difficulties,
    the bigger network should lose learning rate more gracefully

    Essentially, the densities of a larger network should have a shallower
    gradient. There is a limit to gradient shallowness, which is the G1
    (intrinsic task difficulty)
    Run [2:4] :  gamma_to_use = [0.5,0.5,0.1]
    Run [1,2,3] : small network
    Run [4] : big network
    Run [1]: gamma_to_use = [0.1, 0.8,0.02]
    """
    """
    Want to plot learning rate vs absolute error. Show that learning rates are
    similar when absolute error high, and that discrepancy only arises later on
    """
    dt_to_use = 1
    num_runs = 400
    num_repeats = 8
    num_tests = 4
    n_mlp = par.nn_fam_mlp(base_dim=[10,5,5,6],
                                data_cardinality=1000)
    gamma_to_use = []
    outs_all = []
    for i in range(num_tests*num_repeats):
        n_mlp.add_nn(nn_dim=[10,20,20,6])
    #gamma_to_use .append([0.02,0.8,0.])
    plt.style.use('ggplot')
    matplotlib.rcParams.update({'font.size': 12})
    gs = gridspec.GridSpec(num_tests, 2, height_ratios=[2]*num_tests,
                           width_ratios = [1,1], hspace=0.4)
    figs = plt.figure(figsize=(4, 8))
    axL = [figs.add_subplot(gs[j,:]) for j in range(num_tests)]
    [axL[j].set_xticklabels('') for j in range(num_tests-1)]
    [axL[j].get_shared_x_axes().join(axL[j], axL[j-1]) for j in range(1,num_tests)]
    [axL[j].get_shared_y_axes().join(axL[j], axL[j-1]) for j in range(1,num_tests)]
    err_vecs = [[] for j in range(num_tests)]
    k_vecs = copy.deepcopy(err_vecs)
    outs_all = [par.output_class() for j in range(num_tests)]
    def runningMeanFast(x, N):
        return np.convolve(x, np.ones((N,))/N)[(N-1):]

    def jth_run(j):
        #jth test, i repeats
        gamma_to_use.append([0.05, 0.7*(j),0.03*j])
        n_mlp.training_run(indices_to_run=range(j,j+num_repeats) ,
                              gamma_=gamma_to_use[j],
                              number_of_runs=num_runs,
                              delta_t=dt_to_use,
                              gamma1il=[0.98,0.02],
                              outs_class_in=outs_all[j],
                              subsample_fraction=1)
        outs_all[j].calc_k_from_run()
        err_vecs[j] = np.reshape(outs_all[j]._list,(-1))
        err_args = err_vecs[j].argsort()
        err_vecs[j] = err_vecs[j][err_args]
        k_vecs[j] = np.reshape(outs_all[j].calced_ks,(-1))
        k_vecs[j] = k_vecs[j][err_args]
    [jth_run(j) for j in range(num_tests)]
    for j in range(num_tests):
        xy = np.vstack([(k_vecs[j]),
                        err_vecs[j]])
        z = gaussian_kde(xy)(xy)

        # Sort the points by density, so that the densest points are plotted last
        idx = z.argsort()
        x, y, z = k_vecs[j][idx], \
        err_vecs[j][idx], z[idx]
        #axL[j].scatter(err_vecs[j],k_vecs[j], edgecolor='', s=3)
        axL[j].plot(err_vecs[0],k_vecs[0],color='k', linestyle = '--')
        if j > 0:
            axL[j].plot(err_vecs[j],runningMeanFast(k_vecs[j],num_repeats*10),color='b',linewidth=2, alpha = 0.5)
    return outs_all, figs, err_vecs, k_vecs

# =============================================================================
#     for i in range(4):
#         #small_gs = []
#         small_ks = np.zeros_like(g_grid)
#         for j in np.arange(len(g_grid)-1):
#             #small_gs.append(np.mean(outs[i].g_list[outs[i].calced_ks < j]))
#             small_ks[j] = np.mean(outs_all[i].calced_ks[(
#                     outs_all[i].g_list < g_grid[j+1]) & (
#                     outs_all[i].g_list > g_grid[j])])
#         plts.append(plt.scatter(g_grid, small_ks))
#         plt.xlabel('G')
#         plt.ylabel('k')
# =============================================================================
    #plt.legend(plts)
    ax = figg.axes
    #ax.set_yscale('log')
    return outs_all, figg

def save_fig_data(fig, fig_name):
    """
    Goes through each axis and each line on the figure. Saves them
    """

    all_axes = [[] for i in range(len(fig.axes))]
    dict_now = {}
    i=0
    for ax in fig.axes:
        i+=1
        j=0
        for line in ax.lines:
            j+=1
            label_now = 'axis_' + str(i) + '__' + 'line_' + str(j) + '__'
            dict_now[label_now + 'x_data'] = line.get_xdata()
            dict_now[label_now + 'y_data'] = line.get_ydata()
        for other in ax.collections:
            j+=1
            label_now = 'axis_' + str(i) + '__' + 'other_' + str(j) + '__'
            dict_now[label_now + 'both_data'] = other.get_offsets()
        if len(ax.containers) > 0:
            j+=1
            for errb in ax.collections:
                num_errbs = len(errb.get_paths())
                label_now = 'axis_' + str(i) + '__' + 'other_' + str(j) + '__'
                mm = [errb.get_paths()[i].vertices for i in range(num_errbs)]
                dict_now[label_now + 'both_data'] = np.concatenate(mm)


    sio.savemat(fig_name,dict_now)



if __name__ == '__main__':
    #outs, figs,m,ms = small_big_compare_training_curves()
    outs, figs = gGrowthFig()
    #outs,figs = k_pred_plot()
    #outs,figs = motivation_plot_new()
    #outs = lr_vs_err()
    #outs, figs, ev, kv = lr_vs_err()
    #figs, outs, tvj = time_to_MSE()
    #figs, outs = motivation_plot()
    #outs, figs = randomness_lr_plot()
    #axs = figs.axes
    #figs_[0].savefig('rough_gGrowth_linear.pdf')
    #outs, figs = opt_N_curve()
