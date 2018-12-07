# raman_etal_2018

setup_mlp.py contains code for setting up a nonlinear feedforward neural network,
with choices of architecture, nonlinearities, initialisation etc.

pieces_new_resub.py makes a class that holds a family of neural networks of different sizes,
and trains them on the same task. Training is in the way described in the paper, with
updates consisting of a priori fixed combinations of task-relevant and task-irrelevant plasticity.


Options exist for

1) Corrupting the true gradient with task-irrelevant plasticity at each timestep
(in which case the set of input-label pairs being trained on is finite) e.g. as in Figure 4 or Figure 6.

2) Corrupting a stochastic gradient, where there are infinite input-label pairs, with inputs drawn from
a Gaussian distribution and labels being the output of the teacher network under said input (as in Figure 2)

It also has a class out_class, which stores info from the training run.

figure_x.py is the code from which figure x was made in the paper. Figure 6 is made up of the output of
 two scripts: Figure_6_linear.py and Figure_6_nonlinear.py.

