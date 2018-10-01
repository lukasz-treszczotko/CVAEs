# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 14:27:04 2018

@author: user
MIXTURE OF GAUSSIANS
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
version = 4
n_samples_each = 20000
n_samples = 2*n_samples_each
c_samples = 5000

pen_param = 0.01
data_dimension = 2
code_size = 2
hidden_size = 100
learning_rate=0.001
num_epochs = 100
batch_size = 50
n_batches = n_samples // batch_size
samples_to_plot = 1000


tf.reset_default_graph()

# GENERATE SAMPLE DATA
mean1 = [-0.5, -0]
cov1 = [[0.52, -0.5], [-0.5, 0.52]]

mean2 = [0.5, 0.5]
cov2 = [[0.52, 0.5], [0.5, 0.52]]

x1, y1 = np.random.multivariate_normal(mean1, cov1, n_samples_each).T
x1 = x1.reshape((-1,1))
y1 = y1.reshape((-1,1))

x2, y2 = np.random.multivariate_normal(mean2, cov2, n_samples_each).T
x2 = x2.reshape((-1,1))
y2 = y2.reshape((-1,1))

data1 = np.concatenate((x1, y1), axis=1)
data2 = np.concatenate((x2, y2), axis=1)

data = np.concatenate((data1, data2), axis=0)
np.random.shuffle(data)



plt.scatter(data[:,0], data[:,1])
plt.show()



    
    
X = tf.placeholder(tf.float32, [None, data_dimension], name="data")

# CREATE ENCODERS for RANDOM VARIABLES

def make_encoder(data, code_size, variable_scope='encoder'):
    with tf.name_scope(variable_scope):
        h_1 = tf.layers.dense(data, hidden_size, activation=tf.nn.elu, name='hidden_1')
        h_2 = tf.layers.dense(h_1, hidden_size, tf.nn.elu, name='hidden_2')
        loc = tf.layers.dense(h_2, code_size, name='loc')
        #scale = 0.05*tf.ones_like(loc)
        scale = 0.01*tf.layers.dense(h_2, 2, tf.nn.softplus, name="scale")
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
        #scale = 0.05*tf.ones_like(loc)
        scale = 0.01*tf.layers.dense(h_2, np.prod(data_dimension), 
                                activation=tf.nn.softplus, 
                                name='scale',
                                use_bias=False)
        return tfd.MultivariateNormalDiag(loc=loc,
                                          scale_diag=scale)
  
make_encoder = tf.make_template('encoder', make_encoder)
make_decoder = tf.make_template('decoder', make_decoder)

loc, scale = make_encoder(X, code_size, variable_scope="encoder")
encoded_distribution = tfd.MultivariateNormalDiag(loc, scale)

prior = make_prior(code_size=code_size)
code = encoded_distribution.sample()

likelihood = make_decoder(code, data_dimension).log_prob(X)
divergence = tfd.kl_divergence(encoded_distribution, prior)
elbo = tf.reduce_mean(likelihood - divergence)
optimize = tf.train.AdamOptimizer(learning_rate).minimize(-elbo)
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


def conditional_sampler(sampled_variables, conditioned_variables, 
                        condition_values, num_samples):
    
    """ sampled variables is a list of indices,
        sampled variables is a list of indices,
        condition_values is a list of values"""
        
    choice = np.random.choice(len(data[:, 0]), num_samples)
    sampled_data = data[choice, :]
    sampled_data[:, conditioned_variables] = condition_values
    with tf.Session() as sess:
        saver.restore(sess, "./version_" + str(version))
        reconstructed_samples = sess.run(reconstructed_version, 
                                         feed_dict={X: sampled_data})
    return reconstructed_samples

print(80*'_')
print("Sampling the conditioned distribution..")
x_0 = -1
print("We condition on X_0 being equal to ", x_0)

reconstructed_samples = conditional_sampler([1], [0], 
                        [x_0], c_samples)

plt.hist(reconstructed_samples[:, 1], bins=40)
plt.show()
print("Reconstructed means: ", np.mean(reconstructed_samples, axis=0))

       


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
 























