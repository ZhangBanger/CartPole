# Source: http://rl-gym-doc.s3-website-us-west-2.amazonaws.com/mlss/lab1.html
# Implementation of Monte-Carlo Expectation Maximization
import gym
import numpy as np
from gym.wrappers.monitoring import Monitor

from evaluation import noisy_evaluation, do_episode
from utils import get_dim_theta, make_policy

# Task settings:
env = gym.make('CartPole-v0')  # Change as needed
env = Monitor(env, 'tmp/cart-pole-monte-carlo-em-1', force=True)
num_steps = 500  # maximum length of episode
# Alg settings:
n_iter = 100  # number of iterations of CEM
batch_size = 25  # number of samples per batch

dim_theta = get_dim_theta(env)

# Initialize mean and variance
theta_mean = np.zeros(dim_theta)
theta_variance = np.ones(dim_theta)

# Now, for the algorithm
for iteration in range(n_iter):
    # Sample parameter vectors
    thetas = np.vstack([np.random.multivariate_normal(theta_mean, np.diag(theta_variance)) for _ in range(batch_size)])
    rewards = [noisy_evaluation(env, theta, num_steps) for theta in thetas]
    # Weight parameters by score
    # Update theta_mean, theta_std
    theta_mean = np.average(thetas, axis=0, weights=rewards)
    theta_variance = np.average((thetas - theta_mean) ** 2, axis=0, weights=rewards)
    if iteration % 10 == 0:
        print("iteration %i. mean f: %8.3g. max f: %8.3g" % (iteration, np.mean(rewards), np.max(rewards)))
        print("theta mean %s \n theta std %s" % (theta_mean, theta_variance))
    do_episode(make_policy(env, theta_mean), env, num_steps, render=True)

env.close()
