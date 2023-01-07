import gym
import numpy as np
import torch
import yaml
import argparse
from tensorboardX import SummaryWriter
import datetime
from utils.logger import  Log
import os
import json
from trainer.density_ucr_trainer_offline import density_ucr_trainer
from trainer.density_ucr_trainer_online import online_trainer
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='hopper')
    parser.add_argument('--ds', type=str, default='medium-expert')
    parser.add_argument('--algo', type=str, default='density-ucr-offline')
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--log-dir', type=str, default='logs')
    parser.add_argument('--model-path',type=str,default="models")
    parser.add_argument('--save', type=bool, default=True)#是否保存模型
    parser.add_argument('--train', type=bool, default=True)
    parser.add_argument('--device', type=str,default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--gpu-no', default=7, type=int)
    """
    调参数
    """
    parser.add_argument('--q-mode', default="min", type=str)
    parser.add_argument('--deterministic-ucb', default=False, action="store_true")
    parser.add_argument('--eval-log', default=False, action="store_true")
    args = parser.parse_known_args()[0]
    print(args)
    """
    根据 benchmark 及任务  确定配置文件路径
    """
    base_dir = os.getcwd()
    configs_path = os.path.join(base_dir,"configs","mujoco", args.ds,args.task+".yaml")
    configs_file = open(configs_path, encoding="utf-8")
    configs = yaml.load(configs_file,Loader=yaml.FullLoader) #配置文件加载
    configs = argparse.Namespace(**configs)
    """
    调参数
    """
    # configs.misc['eval_steps'] = 500
    configs.actor_critic['q_mode'] = args.q_mode
    configs.offline['eval_log'] = args.eval_log
    configs.vae['vae_model_path'] = base_dir + configs.vae['vae_model_path']
    configs.online['policy_model_path'] = base_dir + configs.online['policy_model_path']
    args.envs = args.task +"-" +args.ds +"-v2"
    print(configs)
    train_envs = gym.make(args.envs)
    eval_envs = gym.make(args.envs)
    args.train_envs = train_envs
    args.state_dim = train_envs.observation_space.shape[0]
    args.action_dim = train_envs.action_space.shape[0] or train_envs.action_space.n
    args.max_action = train_envs.action_space.high[0]
    args.min_action = train_envs.action_space.low[0]
    args.action_space = train_envs.action_space
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    eval_envs.seed(args.seed)
    print("envs name", args.envs)
    print("Observations shape:", args.state_dim)
    print("Actions shape:", args.action_dim)
    print("Action range:", np.min(train_envs.action_space.low),
          np.max(train_envs.action_space.high))
    if args.device == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_no)
        torch.cuda.set_device(args.gpu_no)
    args.writer = None
    args.logger = None
    # 训练模式下才会记录log
    if args.train:
        # log相关
        t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
        args.log_file = f'seed_{args.seed}_{t0}'
        # log path
        log_path = os.path.join(base_dir, "results", args.log_dir, args.envs, args.algo, args.log_file)
        logger = Log(log_path)
        print("log_path", log_path)
        # model path
        args.model_path = os.path.join(base_dir, "results", args.model_path, args.envs, args.algo, args.log_file)
        print("model_path", args.model_path)
        folder = os.path.exists(args.model_path)
        if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
            os.makedirs(args.model_path)
        writer = SummaryWriter(log_path)
        # 写入 yaml 文件
        with open(os.path.join(log_path,args.envs+ ".yaml"), "w") as yaml_file:
            yaml.dump(configs, yaml_file)
        args_json  = parser.parse_known_args()[0]
        with open(os.path.join(log_path, 'args.json'), 'w') as f:
            json.dump(vars(args_json), f, sort_keys=True, indent=4)
        args.writer = writer
        args.logger = logger
    if args.algo == "density-ucr-offline":
        density_ucr_trainer(args, configs,train_envs, eval_envs)
    elif args.algo == "density-ucr-online":
        online_trainer(args, configs,train_envs, eval_envs)

if __name__ == "__main__":
    main()