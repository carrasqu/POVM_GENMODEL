########## Multinomial RBM class ##########
### Juan Carrasquilla
#####################################################################################

import tensorflow as tf
import itertools as it
import numpy as np

class RBM(object):
  
  ### Constructor ###
  def __init__(self, num_hidden, num_visible, num_state_vis, num_state_hid, num_samples=128, weights=None, visible_bias=None, hidden_bias=None):
    self.num_hidden = num_hidden   #number of hidden units
    self.num_state_hid = num_state_hid # number of states of the hidden units
    self.num_visible = num_visible #number of visible units
    self.num_state_vis=num_state_vis # number of states of the visible units
    self.num_samples=num_samples # number of parallel MC chains

    #visible bias:
    default = tf.zeros(shape=(self.num_visible*self.num_state_vis, 1))
    self.visible_bias = self._create_parameter_variable(visible_bias, default)
    
    #hidden bias:
    default = tf.zeros(shape=(self.num_hidden*self.num_state_hid, 1))
    self.hidden_bias = self._create_parameter_variable(hidden_bias, default)
    
    #pairwise weights:
    default = tf.random_normal(shape=(self.num_visible*self.num_state_vis, self.num_hidden*self.num_state_hid), mean=0, stddev=0.05)
    self.weights = self._create_parameter_variable(weights, default)

    #variables for sampling (num_samples is the number of samples to return):
   
       
    #self.hidden_samples = tf.Variable(
    #  self.sample_binary_tensor(tf.constant(0.5), num_samples, self.num_hidden),
    #  trainable=False, name='hidden_samples'
    #)


    self.hidden_samples = tf.Variable( 
      tf.reshape(tf.one_hot(tf.random_uniform([self.num_hidden*num_samples],0,self.num_state_hid,dtype=tf.int32),self.num_state_hid),
      [num_samples, self.num_state_hid*self.num_hidden] ), trainable=False, name='hidden_samples'     
    ) 
    
    self.visible_samples = tf.Variable(
      tf.reshape(tf.one_hot(tf.random_uniform([self.num_visible*num_samples],0,self.num_state_vis,dtype=tf.int32),self.num_state_vis),
      [num_samples, self.num_state_vis*self.num_visible] ), trainable=False, name='visible_samples'
    )    
  


  #end of constructor

  ### Method to initialize variables: ###
  @staticmethod
  def _create_parameter_variable(initial_value=None, default=None):
    if initial_value is None:
      initial_value = default
    return tf.Variable(initial_value)

  ### Method to calculate the logits of conditional probability of the hidden layer given a visible state: ###
  def logits_p_of_h_given(self, v):
    # type: (tf.Tensor) -> tf.Tensor
    #return tf.nn.sigmoid(tf.matmul(v, self.weights) + tf.transpose(self.hidden_bias))
    return tf.matmul(v, self.weights) + tf.transpose(self.hidden_bias)

  ### Method to calculate the logits for conditional probability of the visible layer given a hidden state: ###
  def logits_p_of_v_given(self, h):
    # type: (tf.Tensor) -> tf.Tensor
    return tf.matmul(h, self.weights, transpose_b=True) + tf.transpose(self.visible_bias) 
    #return tf.nn.sigmoid(tf.matmul(h, self.weights, transpose_b=True) + tf.transpose(self.visible_bias))

  ### Method to sample the hidden nodes given a visible state: ###
  def sample_h_given(self, v):
    # type: (tf.Tensor) -> (tf.Tensor, tf.Tensor)
    b = tf.shape(v)[0]  # number of samples
    #m = self.num_hidden
    #prob_h = self.p_of_h_given(v)
    #samples = self.sample_binary_tensor(prob_h, b, m)
    logits_prob_h = self.logits_p_of_h_given(v)
    indices = tf.multinomial(tf.reshape( logits_prob_h,[b*self.num_hidden,self.num_state_hid]),1)  
    samples = tf.reshape(tf.one_hot(indices,depth=self.num_state_hid,axis=1, dtype=tf.float32),[b,self.num_state_hid*self.num_hidden])  
    return samples
  
  ### Method to sample the visible nodes given a hidden state: ###
  def sample_v_given(self, h):
    # type: (tf.Tensor) -> (tf.Tensor, tf.Tensor)
    b = tf.shape(h)[0]  # number rof samples
    #n = self.num_visible
    #prob_v = self.p_of_v_given(h)
    #samples = self.sample_binary_tensor(prob_v, b, n)
    logits_prob_v = self.logits_p_of_v_given(h)
    indices = tf.multinomial(tf.reshape( logits_prob_v,[b*self.num_visible,self.num_state_vis]),1)
    samples = tf.reshape(tf.one_hot(indices,depth=self.num_state_vis,axis=1, dtype=tf.float32),[b,self.num_state_vis*self.num_visible]) 
    return samples
    #logshaped = tf.reshape( logits_prob_v,[nsamples*L,K])
    #indices = tf.multinomial(logshaped,1)
    #onehots = tf.one_hot(indices,depth=K,axis=1, dtype=tf.float32)
    #samples = tf.reshape(onehots, [nsamples,K*L])

  ###
  # Method for persistent contrastive divergence (CD_k):
  # Stores the results of `num_iterations` of contrastive divergence in class variables.
  #
  # :param int num_iterations: The 'k' in CD_k.
  ###
  def stochastic_maximum_likelihood(self, num_iterations,cdimages):
    # type: (int) -> (tf.Tensor, tf.Tensor, tf.Tensor)
      """
      Define persistent CD_k. Stores the results of `num_iterations` of contrastive divergence in
      class variables.
      :param int num_iterations: The 'k' in CD_k.
      """
      if cdimages is None:
          v_samples=self.visible_samples
      else:
          v_samples=cdimages

      h_samples = None
      for i in range(num_iterations):
          h_samples = self.sample_h_given(v_samples)
          v_samples = self.sample_v_given(h_samples)

      self.hidden_samples = self.hidden_samples.assign(h_samples)
      self.visible_samples = self.visible_samples.assign(v_samples)

      return self.hidden_samples, self.visible_samples    

#      h_samples = self.hidden_samples
#      v_samples = None
#      for i in range(num_iterations):
#        v_samples = self.sample_v_given(h_samples)
#        h_samples = self.sample_h_given(v_samples)
#
#      self.hidden_samples = self.hidden_samples.assign(h_samples)
#      return self.hidden_samples, v_samples
 
  ###
  # Method to compute the energy E = - aT*v - bT*h - vT*W*h
  # Note that since we want to support larger batch sizes, we do element-wise multiplication between
  # vT*W and h, and sum along the columns to get a Tensor of shape batch_size by 1
  #
  # :param hidden_samples:  Tensor of shape batch_size by num_hidden
  # :param visible_samples: Tensor of shape batch_size by num_visible
  ###
  def energy(self, hidden_samples, visible_samples):
      # type: (tf.Tensor, tf.Tensor) -> tf.Tensor
      return (-tf.matmul(hidden_samples, self.hidden_bias)  # b x m * m x 1
              - tf.matmul(visible_samples, self.visible_bias)  # b x n * n x 1
              - tf.reduce_sum(tf.matmul(visible_samples, self.weights) * hidden_samples, 1))
 
  def neg_log_likelihood_grad(self, visible_samples, model_samples=None, cdimages=None, num_gibbs=2):
      # type: (tf.Tensor, tf.Tensor, int) -> tf.Tensor

      hidden_samples = self.sample_h_given(visible_samples)
      expectation_from_data = tf.reduce_mean(self.energy(hidden_samples, visible_samples))
      model_hidden, model_visible = self.stochastic_maximum_likelihood(num_gibbs,cdimages)
      expectation_from_model = tf.reduce_mean(self.energy(model_hidden, model_visible))

      return expectation_from_data - expectation_from_model 
  @staticmethod
  def sample_binary_tensor(prob, m, n):
    # type: (tf.Tensor, int, int) -> tf.Tensor
    return tf.where(
      tf.less(tf.random_uniform(shape=(m, n)), prob),
      tf.ones(shape=(m, n)),
      tf.zeros(shape=(m, n))
    )
  def free_energy(self, visible_samples):
      f1 = tf.matmul(visible_samples, self.visible_bias)
      Xbjl = tf.matmul(visible_samples, self.weights)+tf.transpose(self.hidden_bias)
      f3 =   tf.reduce_logsumexp( tf.reshape(Xbjl, [-1,self.num_hidden,self.num_state_hid]), axis=[2])
      free_energy = f1 + tf.reduce_sum(f3, [1], keep_dims=True )   

      return free_energy


#end of RBM class
