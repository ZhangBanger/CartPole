from gym.spaces import Discrete, Box

from policy import DeterministicDiscreteActionLinearPolicy, DeterministicContinuousActionLinearPolicy


def make_policy(environment, theta):
    if isinstance(environment.action_space, Discrete):
        return DeterministicDiscreteActionLinearPolicy(theta,
                                                       environment.observation_space,
                                                       environment.action_space)
    elif isinstance(environment.action_space, Box):
        return DeterministicContinuousActionLinearPolicy(theta,
                                                         environment.observation_space,
                                                         environment.action_space)
    else:
        raise NotImplementedError


def get_dim_theta(env):
    if isinstance(env.action_space, Discrete):
        return (env.observation_space.shape[0] + 1) * env.action_space.n
    elif isinstance(env.action_space, Box):
        return (env.observation_space.shape[0] + 1) * env.action_space.shape[0]
    else:
        raise NotImplementedError
