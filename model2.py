# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:27:04 2018

@author: user
"""

# A general autoencoder model that can be easily tuned to serve different
# purposes



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp
tfd = tfp.distributions




import time


# Hyperparamters:
version = 1
n_samples = 3000


data_dimension = 2
code_size = 2
hidden_size = 200
learning_rate=0.001
num_epochs = 40
batch_size = 100
n_batches = n_samples // batch_size
samples_to_plot = 200



tf.reset_default_graph()

# GENERATE SAMPLE DATA
mean = [-0.5, 0.5]
cov = [[1, -0.5], [-0.5, 1]]
x, y = np.random.multivariate_normal(mean, cov, n_samples).T
x = x.reshape((-1,1))
y = y.reshape((-1,1))

data = np.concatenate((x, y), axis=1)

plt.scatter(data[:,0], data[:,1])
plt.show()



    
    
X = tf.placeholder(tf.float32, [None, data_dimension], name="data")

# a numpy array
#dataset = tf.contrib.data.Dataset.from_tensor_slices(data)
#print(dataset.output_shapes)
#dataset.batch(batch_size)
#print(dataset.output_shapes)
#print(dataset.output_types)

# CREATE ENCODERS for RANDOM VARIABLES


    
    


def make_encoder(data, code_size, variable_scope='encoder'):
    with tf.name_scope(variable_scope):
        h_1 = tf.layers.dense(data, hidden_size, activation=tf.nn.relu, name='hidden_1')
        h_2 = tf.layers.dense(h_1, hidden_size, tf.nn.relu, name='hidden_2')
        loc = tf.layers.dense(h_2, code_size, name='loc')
        scale = 0.1*tf.ones_like(loc)
        #scale = tf.layers.dense(h_2, 2, tf.nn.softplus, name="scale")
        return loc, scale

def make_prior(code_size):
    with tf.variable_scope('prior'):
        loc = tf.zeros(code_size)
        scale = tf.ones(code_size)
        return tfd.MultivariateNormalDiag(loc, scale)
    
def make_decoder(code, data_dimension, variable_scope='decoder'):
    with tf.name_scope(variable_scope):
        
        h_1 = tf.layers.dense(code, hidden_size, tf.nn.relu, name='hidden_1')
        h_2 = tf.layers.dense(h_1, hidden_size, tf.nn.relu, name='hidden_2')
        
        loc = tf.layers.dense(h_2, np.prod(data_dimension), name='loc')
        scale = 0.1*tf.ones_like(loc)
        #scale = tf.layers.dense(h_2, np.prod(data_dimension), 
                                #activation=tf.nn.softplus, 
                                #name='scale',
                                #use_bias=False)
        return tfd.MultivariateNormalDiag(loc=loc,
                                          scale_diag=scale)
  
make_encoder = tf.make_template('encoder', make_encoder)
make_decoder = tf.make_template('decoder', make_decoder)

cum_loc = []   
cum_scale = []     
                                    
for j in range(data_dimension):
        X_j = tf.reshape(X[:, j], (-1,1))
        loc, scale = make_encoder(X_j, code_size, variable_scope="encoder_"+str(j))
        cum_loc.append(loc)
        cum_scale.append(scale)
        
stack_of_loc = tf.stack(cum_loc, axis=1)
stack_of_scale = tf.stack(cum_scale, axis=1)
loc = tf.reduce_mean(stack_of_loc, axis=1)
stack_of_scale_squared = tf.square(stack_of_scale)
scale = tf.reduce_mean(stack_of_scale_squared, axis=1)
divisor = tf.constant(np.sqrt(data_dimension).astype(np.float32))
scale = 0.1*tf.ones_like(loc)

penalty_sum = 0.0
for j in range(data_dimension):
    penalty_sum += tf.square(stack_of_loc[:, j, :] - loc)
penalty_sum = penalty_sum/data_dimension
penalty_sum = tf.reduce_mean(penalty_sum, axis=0)
#scale = tf.truediv(scale, divisor)
#scale = tf.sqrt(scale)

encoded_distribution = tfd.MultivariateNormalDiag(loc, scale)
prior = make_prior(code_size=code_size)
code = encoded_distribution.sample()
likelihood = make_decoder(code, data_dimension).log_prob(X)
divergence = tfd.kl_divergence(encoded_distribution, prior)
elbo = tf.reduce_mean(likelihood - divergence)
optimize = tf.train.AdamOptimizer(learning_rate).minimize(-elbo+penalty_sum)
reconstructed_version = make_decoder(code, data_dimension).mean()

                                       
init = tf.global_variables_initializer()


print('Version: %d' % (version))
print('Training the network... ')
print()
with tf.Session() as sess:
    saver = tf.train.Saver()    
    start_global_time = time.time()
    sess.run(init)
    writer = tf.summary.FileWriter('graphs/alpha_version', sess.graph)
    for epoch in range(num_epochs):
        for step in range(n_batches):
            X_batch = data[(step) * batch_size: (step) * batch_size + batch_size]
            _, train_elbo = sess.run([optimize, elbo], feed_dict=
                                        {X: X_batch})
        if (epoch + 1) % 2 == 0:
            time_now = time.time()
            time_per_epoch = (time_now-start_global_time)/(epoch+1)
            train_elbo = sess.run([elbo], feed_dict={X: X_batch})
            print('Epoch nr: ', epoch+1, ' Current loss: ', train_elbo)
            
            print('Expected time remaining: %.2f seconds.' % (time_per_epoch * (num_epochs - epoch)))
            print(80*'_')
            
            print('Plotting a reconstructed sample...')
            
            reconstruction_samples = sess.run(reconstructed_version, feed_dict={X: X_batch})
            reconstruction_samples = np.reshape(reconstruction_samples, (data_dimension, -1))
            #reconstruction_sample = sess.run(reconstructed_sample[choice, :], feed_dict={X: X_batch})
            original_samples = X_batch
            original_samples = np.reshape(original_samples, (data_dimension, -1))
            plt.figure(figsize=(14,8))
            
            data_plots = (reconstruction_samples, original_samples)
            groups = ("reconstructed", "original")
            colors = ("red", "green")
            fig = plt.figure()
            plt.xlim(-4,4)
            plt.ylim(-4,4)
            ax = fig.add_subplot(1, 1, 1)
            for datap, color, group in zip(data_plots, colors, groups):
                x, y = datap
                ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
            
            plt.show()
            print(80*'_')
            
            print('Plotting the code on the latent_space...')
            choice2 = np.random.randint(n_batches)
            latent = sess.run(encoded_distribution.sample(), feed_dict=
                {X:data[choice2 * batch_size: choice2 * batch_size + batch_size, :]})
            
            
            plt.figure(figsize=(8,8))
            plt.xlim(-4,4)
            plt.ylim(-4,4)
            colors = (0,0,0)
            plt.scatter(latent[:, 0], latent[:, 1], c=colors, alpha=0.5)
            plt.show()
            
            
        
            
        saver.save(sess, "./version_" + str(version))
    writer.close()

def sample_generator(num_samples=200):
    samples = []
    with tf.Session() as sess:
        
        
        saver.restore(sess, "./version_" + str(version))
        for j in range(num_samples):
            code_sample = sess.run(prior.sample())
            code_sample = np.expand_dims(code_sample, axis=0)
            generated_sample = sess.run(reconstructed_version, 
                                    feed_dict={code: code_sample})
            samples.append(np.squeeze(generated_sample))
    return np.array(samples)


def conditional_sampler(sampled_variable, conditioned_variable, condition_value):
    with tf.Session() as sess:
        saver.restore(sess, "./version_" + str(version))
        feed_value = np.zeros((1,2))
        feed_value[0][conditioned_variable] = condition_value
        code_loc, code_scale = sess.run(
                [stack_of_loc[:, conditioned_variable, :], 
                 stack_of_scale[:, conditioned_variable, :]], 
                 feed_dict={X: feed_value})
        reconstruction = sess.run()
        

samples = sample_generator(samples_to_plot)
#samples = np.reshape(samples, (data_dimension, -1))
originals = data[:samples_to_plot, :]
#originals = np.reshape(originals, (data_dimension, -1))

print(80*'_')
print("Plotting the generated samples..")

data_plots = (samples, originals)
groups = ("black_box", "original")
colors = ("red", "blue")



plt.xlim(-3,3)
plt.ylim(-3,3)
plt.scatter(originals[:,0], originals[:,1], c='blue')
plt.show()

plt.xlim(-3,3)
plt.ylim(-3,3)
plt.scatter(samples[:,0], samples[:,1], c='red')
plt.show()
#ax = fig.add_subplot(1, 1, 1)
#for data, color, group in zip(data_plots, colors, groups):
    #x, y = data
    #print(x.shape)
    #ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)
#plt.show()
           
print(80*'_')
print("Calculate empirical covariance matrix..")
emp_cov = np.cov(samples.T)
print(emp_cov)

print(80*'_')
print("Sampling the conditioned distribution..")



















