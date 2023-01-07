import argparse
import numpy as np
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import gym
from tqdm import tqdm
import os
from algo.vae import VAE
import time
from coolname import generate_slug
import utils
import json
import d4rl
from utils.log import Logger
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
def make_dir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        pass
    return dir_path
def snapshot_src(src, target, exclude_from):
    make_dir(target)
    os.system(f"rsync -rv --exclude-from={exclude_from} {src} {target}")
def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )

    def convert_D4RL(self, dataset):
        self.state = dataset['observations']
        self.action = dataset['actions']
        self.next_state = dataset['next_observations']
        self.reward = dataset['rewards'].reshape(-1, 1)
        self.not_done = 1. - dataset['terminals'].reshape(-1, 1)
        self.size = self.state.shape[0]

    def convert_D4RL_finetune(self, dataset):
        self.ptr = dataset['observations'].shape[0]
        self.size = dataset['observations'].shape[0]
        self.state[:self.ptr] = dataset['observations']
        self.action[:self.ptr] = dataset['actions']
        self.next_state[:self.ptr] = dataset['next_observations']
        self.reward[:self.ptr] = dataset['rewards'].reshape(-1, 1)
        self.not_done[:self.ptr] = 1. - dataset['terminals'].reshape(-1, 1)

    def normalize_states(self, eps=1e-3):
        mean = self.state.mean(0, keepdims=True)
        std = self.state.std(0, keepdims=True) + eps
        self.state = (self.state - mean) / std
        self.next_state = (self.next_state - mean) / std
        return mean, std

    def clip_to_eps(self, eps=1e-5):
        lim = 1 - eps
        self.action = np.clip(self.action, -lim, lim)
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=0)
# dataset
parser.add_argument('--task', type=str, default='halfcheetah')
parser.add_argument('--ds', type=str, default='medium')  # medium, medium-replay, medium-expert, expert
parser.add_argument('--version', type=str, default='v2')
# model
parser.add_argument('--model', default='VAE', type=str)
parser.add_argument('--hidden_dim', type=int, default=750)
parser.add_argument('--beta', type=float, default=0.5)
# train
parser.add_argument('--num_iters', type=int, default=int(1e5))
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--scheduler', default=False, action='store_true')
parser.add_argument('--gamma', default=0.95, type=float)
parser.add_argument('--no_max_action', default=False, action='store_true')
parser.add_argument('--clip_to_eps', default=False, action='store_true')
parser.add_argument('--eps', default=1e-4, type=float)
parser.add_argument('--latent_dim', default=None, type=int, help="default: action_dim * 2")
parser.add_argument('--no_normalize', default=False, action='store_true', help="do not normalize states")

parser.add_argument('--eval_data', default=0.0, type=float, help="proportion of data used for validation, e.g. 0.05")
# work dir
parser.add_argument('--work_dir', type=str, default='results')
parser.add_argument('--notes', default=None, type=str)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--gpu-no', default=0, type=int)

args = parser.parse_args()
if args.device == 'cuda':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_no)
    torch.cuda.set_device(args.gpu_no)
# make directory
base_dir = os.getcwd()
make_dir(os.path.join(base_dir, args.work_dir))
args.work_dir +="/train_vae"
make_dir(os.path.join(base_dir, args.work_dir))
work_dir = os.path.join(base_dir, args.work_dir, args.task + '_' + args.ds)
make_dir(args.work_dir)

ts = time.gmtime()
ts = time.strftime("%m-%d-%H:%M", ts)
exp_name = str(args.task) + '-' + str(args.ds) + '-' + ts + '-bs'  \
    + str(args.batch_size) + '-s' + str(args.seed) + '-b' + str(args.beta) + \
    '-h' + str(args.hidden_dim) + '-lr' + str(args.lr) + '-wd' + str(args.weight_decay)
exp_name += '-' + generate_slug(2)
if args.notes is not None:
    exp_name = args.notes + '_' + exp_name
args.work_dir = args.work_dir + '/' + exp_name
make_dir(args.work_dir)

args.model_dir = os.path.join(args.work_dir, 'model')
make_dir(args.model_dir)

with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
    json.dump(vars(args), f, sort_keys=True, indent=4)

# snapshot_src('.', os.path.join(args.work_dir, 'src'), '.gitignore')
logger = Logger(args.work_dir, use_tb=True)

set_seed_everywhere(args.seed)


# load data
env_name = f"{args.task}-{args.ds}-{args.version}"
env = gym.make(env_name)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])
if args.no_max_action:
    max_action = None
print(state_dim, action_dim, max_action)
latent_dim = action_dim * 2
if args.latent_dim is not None:
    latent_dim = args.latent_dim

replay_buffer = ReplayBuffer(state_dim, action_dim)
replay_buffer.convert_D4RL(d4rl.qlearning_dataset(env))
if not args.no_normalize:
    mean, std = replay_buffer.normalize_states()
else:
    print("No normalize")
if args.clip_to_eps:
    replay_buffer.clip_to_eps(args.eps)
states = replay_buffer.state
actions = replay_buffer.action

if args.eval_data:
    eval_size = int(states.shape[0] * args.eval_data)
    eval_idx = np.random.choice(states.shape[0], eval_size, replace=False)
    train_idx = np.setdiff1d(np.arange(states.shape[0]), eval_idx)
    eval_states = states[eval_idx]
    eval_actions = actions[eval_idx]
    states = states[train_idx]
    actions = actions[train_idx]
else:
    eval_states = None
    eval_actions = None

# train
if args.model == 'VAE':
     vae = VAE(state_dim, action_dim, latent_dim, max_action, hidden_dim=args.hidden_dim,device=args.device).to(args.device)
else:
    raise NotImplementedError
optimizer = torch.optim.Adam(vae.parameters(), lr=args.lr, weight_decay=args.weight_decay)
if args.scheduler:
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=args.gamma)

total_size = states.shape[0]
batch_size = args.batch_size

for step in tqdm(range(args.num_iters + 1), desc='train'):
    idx = np.random.choice(total_size, batch_size)
    train_states = torch.from_numpy(states[idx]).to(args.device)
    train_actions = torch.from_numpy(actions[idx]).to(args.device)

    # Variational Auto-Encoder Training
    #训练VAE
    recon, mean, std = vae(train_states, train_actions)
    #重构损失
    recon_loss = F.mse_loss(recon, train_actions)
    #KL散度
    KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
    vae_loss = recon_loss + args.beta * KL_loss

    logger.log('train/recon_loss', recon_loss, step=step)
    logger.log('train/KL_loss', KL_loss, step=step)
    logger.log('train/vae_loss', vae_loss, step=step)

    optimizer.zero_grad()
    vae_loss.backward()
    optimizer.step()

    if step % 5000 == 0:
        logger.dump(step)
        torch.save(vae.state_dict(), '%s/vae_model_%s_%s_b%s_%s.pt' %
                   (args.model_dir, args.task, args.ds, str(args.beta), step))

        if eval_states is not None and eval_actions is not None:
            # vae.eval()
            with torch.no_grad():
                eval_states_tensor = torch.from_numpy(eval_states).to(args.device)
                eval_actions_tensor = torch.from_numpy(eval_actions).to(args.device)

                # Variational Auto-Encoder Evaluation
                recon, mean, std = vae(eval_states_tensor, eval_actions_tensor)

                recon_loss = F.mse_loss(recon, eval_actions_tensor)
                KL_loss = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
                vae_loss = recon_loss + args.beta * KL_loss

                logger.log('eval/recon_loss', recon_loss, step=step)
                logger.log('eval/KL_loss', KL_loss, step=step)
                logger.log('eval/vae_loss', vae_loss, step=step)
            vae.train()

    if args.scheduler and (step + 1) % 10000 == 0:
        logger.log('train/lr', get_lr(optimizer), step=step)
        scheduler.step()

logger._sw.close()
