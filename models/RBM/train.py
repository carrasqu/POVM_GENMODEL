########## Multinomial RBM class  ##########
### Juan Carrasquilla
###########################################

from __future__ import print_function
import tensorflow as tf
import itertools as it
from rbm import RBM
import numpy as np
import math
import os
import sys 
# Input parameters:
p = input('p ')
L                   = 3         # number of qubits
#p                   = 0.01      # noise value 
num_state_vis       = 4         # local size of "Hilbert" space (K = 4 for the tetrahedral POVM)
num_state_hid       = 2         # number of states per hidden bit (binary)
num_visible         = L         # number of visible nodes
num_hidden          = 8         # number of hidden nodes
nsteps              = 360000    # number of training steps
learning_rate_start = 1e-3      # the learning rate will start at this value and decay exponentially
bsize               = 100       # batch size
num_gibbs           = 2         # number of Gibbs iterations (steps of contrastive divergence)
num_samples         = 100       # number of chains in PCD/CD
CD                  = True      # Training algorithm CD = True trains using CD_k with k= num_gibbs CD = False uses PCD_k with k= num_gibbs  



### Function to save weights and biases to a parameter file ###
def save_parameters(sess, rbm,epoch):
    weights, visible_bias, hidden_bias = sess.run([rbm.weights, rbm.visible_bias, rbm.hidden_bias])
    
    parameter_dir = 'RBM_parameters'
    if not(os.path.isdir(parameter_dir)):
      os.mkdir(parameter_dir)
    parameter_file_path =  '%s/parameters_nH%d_L%d' %(parameter_dir,num_hidden,L)
    parameter_file_path += '_p' + str(p)
    parameter_file_path += '_epoch' + str(epoch)
    np.savez_compressed(parameter_file_path, weights=weights, visible_bias=visible_bias, hidden_bias=hidden_bias)

class Placeholders(object):
    pass

class Ops(object):
    pass

weights      = None  #weights None gets them initialized randomly
visible_bias = None  #visible bias None gets them initialized at zero
hidden_bias  = None  #hidden bias None gets them initialized at zero

# Load the MC configuration training data:
trainFileName = 'data/train.txt'
xtrain        = np.loadtxt(trainFileName)
ept           = np.random.permutation(xtrain) # random permutation of training data
iterations_per_epoch = xtrain.shape[0] / bsize  

# Initialize the RBM 
rbm = RBM(num_hidden=num_hidden, num_visible=num_visible,num_state_vis=num_state_vis, num_state_hid=num_state_hid, weights=weights, visible_bias=visible_bias,hidden_bias=hidden_bias, num_samples=num_samples) 

# Initialize operations and placeholders classes
ops          = Ops()
placeholders = Placeholders()
placeholders.visible_samples = tf.placeholder(tf.float32, shape=(None, num_visible*num_state_vis), name='v') # placeholder for training data

total_iterations = 0 # starts at zero 
ops.global_step  = tf.Variable(total_iterations, name='global_step_count', trainable=False)
learning_rate    = tf.train.exponential_decay(
    learning_rate_start,
    ops.global_step,
    100 * xtrain.shape[0]/bsize,
    1.0 # decay rate = 1 means no decay
)
 
if CD==True:
    cost  = rbm.neg_log_likelihood_grad(placeholders.visible_samples,placeholders.visible_samples, num_gibbs=num_gibbs)
    bsize = rbm.num_samples
else: # performs PCD 
    cost  = rbm.neg_log_likelihood_grad(placeholders.visible_samples, num_gibbs=num_gibbs)

   
optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-2)
ops.lr    = learning_rate
ops.train = optimizer.minimize(cost, global_step=ops.global_step)
ops.init  = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())



with tf.Session() as sess:
  sess.run(ops.init)
  
  bcount      = 0  #counter
  epochs_done = 1  #epochs counter
  #sys.exit()
  for ii in range(nsteps):
    if bcount*bsize+ bsize>=xtrain.shape[0]:
      bcount = 0
      ept    = np.random.permutation(xtrain)

    batch     =  ept[ bcount*bsize: bcount*bsize+ bsize,:]
    bcount    += 1
    feed_dict =  {placeholders.visible_samples: batch}
    
    _, num_steps = sess.run([ops.train, ops.global_step], feed_dict=feed_dict)

    if num_steps % iterations_per_epoch == 0:
      print ('Epoch = %d' % epochs_done)
      save_parameters(sess, rbm,epochs_done)
      epochs_done += 1
