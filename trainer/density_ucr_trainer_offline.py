
from utils.trainer_utils import evaluate_d4rl
from algo.buffer.replay_buffer import ReplayBuffer
import d4rl
import torch
from algo.density_ucr import DensityUCR
def density_ucr_trainer(args,configs,train_envs,eval_envs):
    dataset = d4rl.qlearning_dataset(train_envs)
    max_eval_avg_reward = 0
    total_steps = 0
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
        random_num_samples = configs.offline['random_num_samples'],
        actor_lr=configs.actor_critic['actor_lr'],
        critic_lr=configs.actor_critic['critic_lr'],
        tau=configs.actor_critic['tau'],
        gamma=configs.actor_critic['gamma'],
        num_critics = configs.actor_critic['critic_num'],
        alpha_lr=configs.actor_critic['alpha_lr'],
        device=args.device,
        q_mode = configs.actor_critic['q_mode'],
        base_weight=configs.offline['base_weight'],
        ucb_ratio=configs.offline['ucb_ratio'],
        eval_log = configs.offline['eval_log']
    )
    offline_replay_buffer = ReplayBuffer(configs.actor_critic['buffer_size'])
    offline_replay_buffer.convert_D4RL(dataset)
    # 开始one-step的训练
    # iterate between eval and improvement
    for step in range(int(configs.offline['n_steps'])):
        batch_list = offline_replay_buffer.sample(configs.actor_critic['batch_size'])
        log_info = agent.learn(batch_list)
        if args.writer != None and log_info != None:
            for k,v in log_info.items():
                args.writer.add_scalar("train/"+k, v, total_steps)
        if step % int(configs.misc['eval_steps']) == 0:
            returns_mean, _, _, _ = evaluate_d4rl(eval_envs, agent, configs.misc['eval_episodes'], total_steps,
                                                  args.writer, args.logger)
            if args.save and (returns_mean > max_eval_avg_reward or step % 50000 ==0):
                max_eval_avg_reward = returns_mean
                torch.save(agent.state_dict(), args.model_path + f"/reward{returns_mean:.0f}.pth")
        total_steps += 1