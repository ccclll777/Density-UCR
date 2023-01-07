from utils.trainer_utils import evaluate_d4rl
from algo.buffer.replay_buffer import ReplayBuffer
from algo.density_urc_online import DensityUCR
from algo.buffer.prioritized_buffer import PrioritizedReplay
import torch
import numpy as np
from tqdm import trange
def offline_buffer_to_online(offline_replay_buffer: ReplayBuffer,online_replay_buffer : PrioritizedReplay,agent:DensityUCR,device):
    n = len(offline_replay_buffer)
    for start in range(0,n,1000):
        end = min(start+1000,n)
        batch_list = offline_replay_buffer.get_transitions(start,end)
        with torch.no_grad():
            state = torch.Tensor(np.array(batch_list["state_list"])).to(device)
            action = torch.Tensor(np.array(batch_list["action_list"])).to(device)
            vae_loss = agent.vae.elbo_loss(state,action,0.5,1)
            prioritys = (torch.sigmoid(-2.0*vae_loss) ).detach().cpu().numpy()
            size = state.shape[0]
            for i in range(size):
                online_replay_buffer.push_with_priority(batch_list["state_list"][i],
                                                        batch_list["action_list"][i],
                                                        batch_list["reward_list"][i],
                                                        batch_list["next_state_list"][i],
                                                        batch_list["done_list"][i],
                                                    prioritys[i])
def transitions_to_online(transition_list,online_replay_buffer : PrioritizedReplay,loss,device):
    state, action, reward, next_state, done = map(np.stack, zip(*transition_list))
    with torch.no_grad():
        state_tensor = torch.Tensor(np.array(state)).to(device)
        action_tensor = torch.Tensor(np.array(action)).to(device)
        vae_loss = loss(state_tensor,action_tensor,0.5,1).detach()
        prioritys = (torch.sigmoid(-2.0*vae_loss) ).detach().cpu().numpy()
        size = state.shape[0]
        for i in range(size):
            online_replay_buffer.push_with_priority(state[i],
                                                        action[i],
                                                        reward[i],
                                                        next_state[i],
                                                        done[i],
                                                        prioritys[i])
def online_trainer(args,configs,train_envs,eval_envs):
    agent = DensityUCR(
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        action_space=args.action_space,
        hidden_width=configs.actor_critic['hidden_width'],
        vae_model_path = configs.vae['vae_model_path'],
        vae_latent_dim = 2 * args.action_dim,
        vae_num_samples = configs.vae['num_samples'],
        env_name=args.envs,
        vae_mask=configs.offline['vae_mask'],
        vae_loss_clip_min=configs.offline['vae_loss_clip_min'],
        vae_loss_clip_max=configs.offline['vae_loss_clip_max'],
        backup_entropy=configs.actor_critic['backup_entropy'],
        random_num_samples = configs.offline['random_num_samples'],
        actor_lr=configs.actor_critic['actor_lr'],
        critic_lr=configs.actor_critic['critic_lr'],
        tau=configs.actor_critic['tau'],
        gamma=configs.actor_critic['gamma'],
        num_critics = configs.actor_critic['critic_num'],
        alpha_lr=configs.actor_critic['alpha_lr'],
        device=args.device,
        q_mode = configs.actor_critic['q_mode'],
        ucb_ratio=configs.offline['ucb_ratio'],
        deterministic_ucb = args.deterministic_ucb)
    agent.load_state_dict(torch.load(configs.online['policy_model_path'],map_location="cpu"))
    dataset = train_envs.get_dataset()
    "online_fine_tuning"
    #新建一个online_buffer 然后吧dataset的数据进行蒸馏，选出Top N的轨迹
    offline_replay_buffer = ReplayBuffer(configs.actor_critic['buffer_size'])
    offline_replay_buffer.distill_D4RL(dataset)
    online_replay_buffer = PrioritizedReplay(capacity=configs.online['buffer_size'])
    #填充online的buffer
    offline_buffer_to_online(offline_replay_buffer,online_replay_buffer,agent,args.device)
    max_eval_avg_reward = 0
    evaluate_step = 0
    state = train_envs.reset()
    done = False
    episode_total_reward = 0
    steps = 0
    episode = 0
    total_steps = 0
    returns_mean, _, _, _ = evaluate_d4rl(eval_envs, agent, configs.misc['eval_episodes'], total_steps,
                                          args.writer, args.logger)
    transition_list = []
    for _ in trange(configs.online['train_steps']):
        action = agent.choose_action(state)
        # Obser reward and next obs
        next_state, reward, done, infos = train_envs.step(action)
        episode_total_reward += reward
        transition_list.append((state, action, reward, next_state, done))
        if len(online_replay_buffer) > configs.actor_critic['batch_size']:
            batch_list = online_replay_buffer.sample(configs.actor_critic['batch_size'])
            agent.learn_online(batch_list)
            #更新优先级
            # VAE拟合数据之后，更新这些数据的优先级
            index = batch_list["indices_list"]
            with torch.no_grad():
                state_batch = torch.Tensor(np.array(batch_list["state_list"])).to(args.device)
                action_batch = torch.Tensor(np.array(batch_list["action_list"])).to(args.device)
                vae_loss = agent.vae.elbo_loss(state_batch, action_batch, 0.5, 1).detach()
                prioritys = (torch.sigmoid(-2.0*vae_loss)).detach().cpu().numpy()
            online_replay_buffer.update_priorities(index, prioritys)
        if args.render:
            train_envs.render()
        steps += 1
        total_steps += 1
        evaluate_step += 1
        state = next_state
        if done or  configs.misc['episode_max_steps'] < steps:
            #episode结束
            # agent.vae.eval()
            transitions_to_online(transition_list,online_replay_buffer,agent.vae.elbo_loss,args.device)
            # agent.vae.train()
            transition_list = []
            state, done = train_envs.reset(), False
            episode_total_reward = 0
            steps = 0
            episode +=1
        """
        评估结果
        """
        if evaluate_step % int(configs.misc['eval_steps'] ) == 0:
            evaluate_step = 0
            # 评估结果
            returns_mean, _, _, _ = evaluate_d4rl(eval_envs, agent, configs.misc['eval_episodes'], total_steps,
                                                      args.writer, args.logger)
            if args.save and returns_mean > max_eval_avg_reward:
                max_eval_avg_reward = returns_mean
                torch.save(agent.state_dict(), args.model_path + f"/reward{returns_mean:.0f}.pth")
