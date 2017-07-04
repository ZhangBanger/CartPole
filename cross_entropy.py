# Source: http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab1.html
import gym
import numpy as np
from gym.wrappers.monitoring import Monitor

from policy import Policy

# Task settings:
env = gym.make('CartPole-v0')  # Change as needed
env = Monitor(env, 'tmp/cart-pole-cross-entropy-1', force=True)
num_steps = 500  # maximum length of episode
# Alg settings:
n_iter = 100  # number of iterations of CEM
batch_size = 25  # number of samples per batch
elite_ratio = 0.2  # fraction of samples used as elite set

dim_theta = Policy.get_dim_theta(env)

# Initialize mean and standard deviation
theta_mean = np.zeros(dim_theta)
theta_std = np.ones(dim_theta)

# Now, for the algorithm
for iteration in range(n_iter):
    # Sample parameter vectors
    thetas = np.vstack([np.random.multivariate_normal(theta_mean, np.diag(theta_std ** 2)) for _ in range(batch_size)])
    rewards = [Policy.make_policy(env, theta).evaluate(env, num_steps) for theta in thetas]
    # Get elite parameters
    n_elite = int(batch_size * elite_ratio)
    elite_indices = np.argsort(rewards)[batch_size - n_elite:batch_size]
    elite_thetas = [thetas[i] for i in elite_indices]
    # Update theta_mean, theta_std
    theta_mean = np.mean(elite_thetas, axis=0)
    theta_std = np.std(elite_thetas, axis=0)
    if iteration % 10 == 0:
        print("iteration %i. mean f: %8.3g. max f: %8.3g" % (iteration, np.mean(rewards), np.max(rewards)))
        print("theta mean %s \n theta std %s" % (theta_mean, theta_std))
        # Demonstrate this policy
        Policy.make_policy(env, theta_mean).evaluate(env, num_steps, render=True)

env.close()
