from utils import make_policy


def do_episode(policy, env, max_steps, render=False):
    total_rew = 0
    ob = env.reset()
    for t in range(max_steps):
        a = policy.act(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if render and t % 3 == 0:
            env.render()
        if done:
            break
    return total_rew


def noisy_evaluation(env, theta, num_steps):
    policy = make_policy(env, theta)
    rew = do_episode(policy, env, num_steps)
    return rew
