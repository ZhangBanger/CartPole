import gym
import numpy as np

env = gym.make('CartPole-v0')
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

best_params = None
best_reward = -np.inf
parameters = np.random.rand(4) * 2 - 1
num_eval_eps = 10

for i_episode in range(10000):
    # Weights are 1x4 matrix
    # µ = 0 , sigma 1
    noise_scaling = 0.1
    print("Applying jitter to parameters with variance {0}".format(noise_scaling))
    new_params = parameters + (np.random.rand(4) * 2 - 1) * noise_scaling
    episodic_reward = evaluate_policy(num_eval_eps, parameters)
    if episodic_reward > best_reward:
        print("Got new best reward of {0}, better than previous of {1}".format(
            episodic_reward,
            best_reward,
        ))
        best_reward = episodic_reward
        best_params = parameters
        # Solution calls for keeping pole up for 200 timesteps
        if episodic_reward >= 200:
            print("Episode {0} attained reward of {1}, terminating".format(i_episode * num_eval_eps, episodic_reward))
            break
