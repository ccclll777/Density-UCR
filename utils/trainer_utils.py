import numpy as np
import torch
# import d3rlpy
def evaluate_d4rl(env, agent, eval_episode,total_steps = 0,writer = None,logger = None,online = False):
    returns = np.zeros(eval_episode)
    for episode in range(eval_episode):
        obs, done = env.reset(), False
        step = 0
        episode_return = 0
        while not done :
            action = agent.choose_action(obs,evaluate=True)
            next_obs, reward, done, info = env.step(action)
            episode_return += reward
            obs = next_obs
            step += 1
        returns[episode] = episode_return
    norm_returns = env.get_normalized_score(returns) * 100
    returns_mean = returns.mean()
    returns_std = returns.std()
    norm_returns_mean = norm_returns.mean()
    norm_returns_std = norm_returns.std()
    if writer != None:
        writer.add_scalar("evaluate_d4rl/return mean", returns_mean, total_steps)
        writer.add_scalar("evaluate_d4rl/return std", returns_std, total_steps)
        writer.add_scalar("evaluate_d4rl/normalized return mean", norm_returns_mean, total_steps)
        writer.add_scalar("evaluate_d4rl/normalized return std", norm_returns_std, total_steps)
    if logger != None:
        logger.row({
            'steps': total_steps,
            'return mean': returns_mean,
            'return std': returns_std,
            'normalized return mean': norm_returns_mean,
            'normalized return std': norm_returns_std,
        })
    else:
        print("evaluate_d4rl/return mean", returns_mean,
              "evaluate_d4rl/return std", returns_std,
              "evaluate_d4rl/normalized return mean", norm_returns_mean,
              "evaluate_d4rl/normalized return std", norm_returns_std,
              "total_steps",total_steps)

    return returns_mean,returns_std,norm_returns_mean,norm_returns_std
def critic_evaluate(env, agent, eval_episodes,total_steps=0,writer = None,logger = None):
    returns = np.zeros(eval_episodes)
    states = []
    actions = []
    for episode in range(eval_episodes):
        obs = env.reset()
        action =  agent.choose_action(obs,evaluate=True)
        states.append(np.expand_dims(obs, 0))
        actions.append(np.expand_dims(action, 0))
        ep_return = 0
        done = False
        step = 0
        while not done:
            next_obs, reward, done, _ = env.step(action)
            obs = next_obs
            ep_return += np.power(agent.gamma, step) * reward
            action = agent.choose_action(obs,evaluate=True)
            step += 1
        returns[episode] = ep_return
    states = np.concatenate(states, axis=0)
    actions = np.concatenate(actions, axis=0)

    preds = agent.critic_predict(states, actions).cpu().detach().numpy()
    mse = np.mean((preds - returns) ** 2)
    if writer != None:
        writer.add_scalar("q/mse", mse, total_steps)
        writer.add_scalar("q/pred_mean",  np.mean(preds), total_steps)
        writer.add_scalar("q/rollout_mean", np.mean(returns), total_steps)
    if logger != None:
        logger.row({
            'q/steps': total_steps,
            'q/mse': mse,
            'q/pred_mean':  np.mean(preds),
            'q/rollout_mean': np.mean(returns),
        })
    print("q/mse", mse, "steps:",total_steps)
    return preds, returns
def sample_batch(dataset, batch_size):
    k = list(dataset.keys())[0]
    n, device = len(dataset[k]), dataset[k].device
    for v in dataset.values():
        assert len(v) == n, 'Dataset values must have same length'
    indices = torch.randint(low=0, high=n, size=(batch_size,), device=device)
    return {k: v[indices] for k, v in dataset.items()}

def evaluate(env, agent, eval_episode, episode_max_steps,total_steps = 0,writer = None):
    avg_reward = 0.
    avg_length = 0
    for _ in range(eval_episode):
        obs, done = env.reset(), False
        step = 0
        while not done and step <= episode_max_steps:
            action = agent.choose_action(obs,evaluate=True)

            next_obs, reward, done, info = env.step(action)
            avg_reward += reward
            obs = next_obs
            step += 1
        avg_length += step
    avg_reward /= eval_episode
    avg_length /= eval_episode
    if writer != None:
        writer.add_scalar("evaluate/reward", avg_reward, total_steps)
        writer.add_scalar("evaluate/length", avg_length, total_steps)

    print("evaluate/step:", total_steps, "steps", "average_reward", avg_reward)
    print("evaluate/step:", total_steps, "steps", "average_length", avg_length)
    return avg_reward,avg_length
