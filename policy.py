# ================================================================
# Policies
# ================================================================
import numpy as np

from gym.spaces import Discrete, Box


class Policy(object):
    def __init__(self):
        pass

    def act(self, obs):
        raise NotImplementedError

    @staticmethod
    def make_policy(env, theta):
        if isinstance(env.action_space, Discrete):
            return DeterministicDiscreteActionLinearPolicy(
                theta,
                env.observation_space,
                env.action_space,
            )
        elif isinstance(env.action_space, Box):
            return DeterministicContinuousActionLinearPolicy(
                theta,
                env.observation_space,
                env.action_space,
            )
        else:
            raise NotImplementedError

    @staticmethod
    def get_dim_theta(env):
        if isinstance(env.action_space, Discrete):
            return (env.observation_space.shape[0] + 1) * env.action_space.n
        elif isinstance(env.action_space, Box):
            return (env.observation_space.shape[0] + 1) * env.action_space.shape[0]
        else:
            raise NotImplementedError

    def evaluate(self, env, num_steps, render=False):
        total_rew = 0
        ob = env.reset()
        for t in range(num_steps):
            a = self.act(ob)
            (ob, reward, done, _info) = env.step(a)
            total_rew += reward
            if render and t % 3 == 0:
                env.render()
            if done:
                break
        return total_rew


class DeterministicDiscreteActionLinearPolicy(Policy):
    def __init__(self, theta, ob_space, ac_space):
        """
        dim_ob: dimension of observations
        n_actions: number of actions
        theta: flat vector of parameters
        """
        Policy.__init__(self)
        dim_ob = ob_space.shape[0]
        n_actions = ac_space.n
        assert len(theta) == (dim_ob + 1) * n_actions
        self.W = theta[0: dim_ob * n_actions].reshape(dim_ob, n_actions)
        self.b = theta[dim_ob * n_actions: None].reshape(1, n_actions)

    def act(self, ob):
        """
        """
        y = ob.dot(self.W) + self.b
        a = y.argmax()
        return a


class DeterministicContinuousActionLinearPolicy(Policy):
    def __init__(self, theta, ob_space, ac_space):
        """
        dim_ob: dimension of observations
        dim_ac: dimension of action vector
        theta: flat vector of parameters
        """
        Policy.__init__(self)
        self.ac_space = ac_space
        dim_ob = ob_space.shape[0]
        dim_ac = ac_space.shape[0]
        assert len(theta) == (dim_ob + 1) * dim_ac
        self.W = theta[0: dim_ob * dim_ac].reshape(dim_ob, dim_ac)
        self.b = theta[dim_ob * dim_ac: None]

    def act(self, ob):
        a = np.clip(ob.dot(self.W) + self.b, self.ac_space.low, self.ac_space.high)
        return a
