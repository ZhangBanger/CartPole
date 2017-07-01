import gym
import numpy as np
from gym.wrappers.monitoring import Monitor

env = gym.make('CartPole-v0')
env = Monitor(env, 'tmp/cart-pole-hill-climb-2', force=True)

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


best_reward = -np.inf
best_params = np.random.rand(4) * 2 - 1
num_eval_eps = 10
base_noise_factor = 1.
num_evaluations = 1000

print("Running Hill Climb on Cart Pole")
print("Params:\n\tMC Eval Count: {0} trajectories\n\tBase Noise Factor: {1}".format(
    num_eval_eps,
    base_noise_factor,
))

for i_episode in range(num_evaluations):
    # Weights are 1x4 matrix
    # µ = 0 , sigma 1
    noise_scaling = 1 - (i_episode / num_evaluations)
    print("Applying jitter with factor {0} to parameters {1}".format(
        noise_scaling,
        best_params,
    ))

    noise_term = (np.random.rand(4) * 2 - 1) * noise_scaling
    parameters = best_params + noise_term
    episodic_reward = evaluate_policy(num_eval_eps, parameters)
    if episodic_reward > best_reward:
        print("Episode {2}: Got new best reward of {0}, better than previous of {1}".format(
            episodic_reward,
            best_reward,
            i_episode,
        ))
        best_reward = episodic_reward
        best_params = parameters

env.close()
