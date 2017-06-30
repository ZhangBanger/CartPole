import gym
import numpy as np

from gym.wrappers.monitoring import Monitor

env = gym.make('CartPole-v0')
env = Monitor(env, 'tmp/cart-pole-random-search-1', force=True)
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


best_params = None
best_reward = -np.inf
completed = 0

for i_episode in range(10000):
    # Weights are 1x4 matrix
    # µ = 0 , sigma 1
    parameters = np.random.rand(4) * 2 - 1
    episodic_reward = run_episode(parameters)

    if episodic_reward > 195:
        completed += 1
        if completed > 100:
            break

    if episodic_reward > best_reward:
        print("Got new best reward of {0}, better than previous of {1}".format(
            episodic_reward,
            best_reward,
        ))
        best_reward = episodic_reward
        best_params = parameters

if completed < 100:
    print("CartPole-v0 unsolved after 10k episodes")

env.close()
