import gym
import numpy as np
from gym.wrappers.monitoring import Monitor

MC_POLICY_EVAL_EP = 10
BASE_NOISE_FACTOR = 0.1
NUM_POLICY_EVAL = 500

env = gym.make('CartPole-v0')
env = Monitor(env, 'tmp/cart-pole-hill-climb-4', force=True)

print("Action space: {0}".format(env.action_space))
print("Observation space: {0}\n\tLow: {1}\n\tHigh: {2}".format(
    env.observation_space,
    env.observation_space.low,
    env.observation_space.high,
))


def action_selection(weights, observation):
    if np.matmul(weights, observation) < 0:
        return 0
    else:
        return 1


def run_episode(weights):
    observation = env.reset()
    total_reward = 0
    for t in range(200):
        env.render()
        action = action_selection(weights, observation)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print("Episode finished after {0} timesteps with reward {1}".format(
                t + 1,
                total_reward,
            ))
            break

    return total_reward


def evaluate_policy(num_episodes, weights):
    mean_reward = 0
    for k in range(1, num_episodes + 1):
        reward = run_episode(weights)
        error = reward - mean_reward
        mean_reward += error / k

    print("Mean reward estimated as {0} for past {1} episodes".format(
        mean_reward,
        num_episodes
    ))
    return mean_reward


last_reward = -np.inf
best_params = np.random.rand(4) * 2 - 1

print("Running Hill Climb on Cart Pole")
print("Params:\n\tMC Eval Count: {0} trajectories\n\tBase Noise Factor: {1}".format(
    MC_POLICY_EVAL_EP,
    BASE_NOISE_FACTOR,
))

for i_episode in range(NUM_POLICY_EVAL):
    # Weights are 1x4 matrix
    # µ = 0 , sigma 1
    annealing_term = 1 - (i_episode / NUM_POLICY_EVAL)
    noise_scaling = BASE_NOISE_FACTOR * annealing_term
    print("Applying jitter with factor {0} to parameters {1}".format(
        noise_scaling,
        best_params,
    ))

    # Add gaussian noise
    # µ = 0 , sigma = noise_scaling
    noise_term = np.random.randn(4) * noise_scaling
    parameters = best_params + noise_term
    episodic_reward = evaluate_policy(MC_POLICY_EVAL_EP, parameters)
    if episodic_reward > last_reward:
        print("Reward {0} is improvement from previous evaluation {1} - Eval {2}".format(
            episodic_reward,
            last_reward,
            i_episode,
        ))
        print("Updating parameters\n\tfrom {0}\n\tto{1}".format(
            best_params,
            parameters
        ))
        best_params = parameters

    last_reward = episodic_reward


env.close()
