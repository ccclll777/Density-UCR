

from copy import deepcopy
import math
from algo.vae import VAE
import numpy as np
import torch
from torch.distributions import Normal
import torch.nn as nn
from utils.schedule import ConstantSchedule
def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)
# SAC Actor & Critic implementation
class VectorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ensemble_size = ensemble_size

        self.weight = nn.Parameter(torch.empty(ensemble_size, in_features, out_features))
        self.bias = nn.Parameter(torch.empty(ensemble_size, 1, out_features))

        self.reset_parameters()

    def reset_parameters(self):
        # default pytorch init for nn.Linear module
        for layer in range(self.ensemble_size):
            nn.init.kaiming_uniform_(self.weight[layer], a=math.sqrt(5))

        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: [ensemble_size, batch_size, input_size]
        # weight: [ensemble_size, input_size, out_size]
        # out: [ensemble_size, batch_size, out_size]
        return x @ self.weight + self.bias


class Actor(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, max_action: float = 1.0
    ):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # with separate layers works better than with Linear(hidden_dim, 2 * action_dim)
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_sigma = nn.Linear(hidden_dim, action_dim)

        # init as in the EDAC paper
        for layer in self.trunk[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.mu.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mu.bias, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.log_sigma.bias, -1e-3, 1e-3)

        self.action_dim = action_dim
        self.max_action = max_action

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        need_log_prob: bool = False,
    ) :
        hidden = self.trunk(state)
        mu, log_sigma = self.mu(hidden), self.log_sigma(hidden)

        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        log_sigma = torch.clip(log_sigma, -5, 2)
        policy_dist = Normal(mu, torch.exp(log_sigma))

        if deterministic:
            action = mu
        else:
            action = policy_dist.rsample()

        tanh_action, log_prob = torch.tanh(action), None
        if need_log_prob:
            # change of variables formula (SAC paper, appendix C, eq 21)
            log_prob = policy_dist.log_prob(action).sum(axis=-1)
            log_prob = log_prob - torch.log(1 - tanh_action.pow(2) + 1e-6).sum(dim=-1)

        return tanh_action * self.max_action, log_prob

    @torch.no_grad()
    def act(self, state: np.ndarray, device: str) -> np.ndarray:
        deterministic = not self.training
        state = torch.tensor(state, device=device, dtype=torch.float32)
        action = self(state, deterministic=deterministic)[0].cpu().numpy()
        return action


class VectorizedCritic(nn.Module):
    def __init__(
        self, state_dim: int, action_dim: int, hidden_dim: int, num_critics: int
    ):
        super().__init__()
        self.critic = nn.Sequential(
            VectorizedLinear(state_dim + action_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, hidden_dim, num_critics),
            nn.ReLU(),
            VectorizedLinear(hidden_dim, 1, num_critics),
        )
        # init as in the EDAC paper
        for layer in self.critic[::2]:
            torch.nn.init.constant_(layer.bias, 0.1)

        torch.nn.init.uniform_(self.critic[-1].weight, -3e-3, 3e-3)
        torch.nn.init.uniform_(self.critic[-1].bias, -3e-3, 3e-3)

        self.num_critics = num_critics

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # [..., batch_size, state_dim + action_dim]
        state_action = torch.cat([state, action], dim=-1)
        if state_action.dim() != 3:
            assert state_action.dim() == 2
            # [num_critics, batch_size, state_dim + action_dim]
            state_action = state_action.unsqueeze(0).repeat_interleave(
                self.num_critics, dim=0
            )
        assert state_action.dim() == 3
        assert state_action.shape[0] == self.num_critics
        # [num_critics, batch_size]
        q_values = self.critic(state_action).squeeze(-1)
        return q_values
class DensityUCR(nn.Module):
    def __init__(
        self,
            state_dim,
            action_dim,
            hidden_width,
            action_space,
            vae_model_path,
            vae_latent_dim,
            vae_num_samples,
            env_name,
            vae_mask=False,
            vae_loss_clip_min = -np.inf,
            vae_loss_clip_max = np.inf,
            backup_entropy = False,
            random_num_samples=10,
            gamma=0.99,
            tau=0.005,
            actor_lr=3e-4,
            critic_lr=3e-4,
            num_critics=10,
            alpha_lr=3e-4,
            device='cpu',
            q_mode='min',
            base_weight = 2.0,
            ucb_ratio=0.01,
            eval_log = False
    ):
        super(DensityUCR, self).__init__()
        self.device = device
        self.action_dim = action_dim
        # Actor & Critic setup
        self.q_mode = q_mode
        max_action = float(action_space.high[0])
        self.actor = Actor(state_dim, action_dim, hidden_width, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic = VectorizedCritic(state_dim, action_dim, hidden_width, num_critics).to(self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )
        with torch.no_grad():
            self.target_critic = deepcopy(self.critic)
        self.tau = tau
        self.gamma = gamma
        self.qf_criterion = nn.MSELoss(reduce=False)
        self.num_critics = num_critics
        self.num_q_update_steps = 0
        # adaptive alpha setup
        self.target_entropy = -float(self.actor.action_dim)
        self.log_alpha = torch.tensor(
            [0.0], dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp().detach()
        self.env_name = env_name
        self.random_num_samples = random_num_samples
        self.vae_mask = vae_mask
        self.ucb_ratio = ucb_ratio
        self.backup_entropy = backup_entropy
        self.eval_log = eval_log
        if self.vae_mask:
            self.vae_num_samples = vae_num_samples
            tmp_vae = VAE(state_dim, action_dim,vae_latent_dim, max_action)
            tmp_vae.load_state_dict(torch.load(vae_model_path,map_location="cpu"))
            self.vae = deepcopy(tmp_vae).to(self.device)
            del tmp_vae
            self.vae.eval()
            if vae_loss_clip_min == '-inf':
                self.vae_loss_clip_min = -np.inf
            else:
                self.vae_loss_clip_min =vae_loss_clip_min
            if vae_loss_clip_max == 'inf':
                self.vae_loss_clip_max = np.inf
            else:
                self.vae_loss_clip_max =vae_loss_clip_max
        self.w_schedule = ConstantSchedule(base_weight)
    def choose_action(self, state, evaluate=False):
        # Return the action to interact with env.
        if len(state.shape) == 1:  # if no batch dim
            state = state.reshape(1, -1)
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).to(self.device)
        else:
            state = state.to(self.device)
        action,_ = self.actor(state,deterministic=evaluate)
        return action.detach().cpu().numpy()[0]
    def alpha_loss(self, state):
        with torch.no_grad():
            action, action_log_prob = self.actor(state, need_log_prob=True)
        loss = (-self.log_alpha * (action_log_prob + self.target_entropy)).mean()
        return loss
    def actor_loss(self,state) :
        pi, action_log_prob = self.actor(state, need_log_prob=True)
        q_value_dist = self.critic(state, pi)
        assert q_value_dist.shape[0] == self.critic.num_critics
        if self.q_mode == 'min':
            q_value = q_value_dist.min(0).values
        elif self.q_mode == 'ave':
            q_value = q_value_dist.mean(0)
        else:
            q_value = q_value_dist.min(0).values
        # needed for logging
        q_value_std = q_value_dist.std(0).mean().item()
        batch_entropy = -action_log_prob.mean().item()
        assert action_log_prob.shape == q_value.shape
        loss = (self.alpha * action_log_prob - q_value).mean()
        return loss, {"batch_entropy":batch_entropy, "q_value_std":q_value_std}
    def ucb_func(self, state_batch, action_batch):
        action_shape = action_batch.shape[0]  # 2560
        state_shape = state_batch.shape[0]  # 256
        num_repeat = int(action_shape / state_shape)  # 10
        if num_repeat != 1:
            state_batch = state_batch.unsqueeze(1).repeat(1, num_repeat, 1).view(state_batch.shape[0] * num_repeat,
                                                                 state_batch.shape[1])  # （2560, obs_dim）
        # Bootstrapped uncertainty
        q_pred = self.critic(state_batch,action_batch)
        ucb = torch.std(q_pred, dim=0, keepdim=True)  # (2560, 1)
        return ucb, q_pred
    def ucb_func_target(self, next_state_batch, next_action_batch):
        # Using the target-Q network to calculate the bootstrapped uncertainty
        # Sample 10 ood actions for each obs, so the obs should be expanded before calculating
        action_shape = next_action_batch.shape[0]  # 2560
        state_shape = next_state_batch.shape[0]  # 256
        num_repeat = int(action_shape / state_shape)  # 10
        if num_repeat != 1:
            next_state_batch = next_state_batch.unsqueeze(1).repeat(1, num_repeat, 1).view(next_state_batch.shape[0] * num_repeat,
                                                                           next_state_batch.shape[1])  # （2560, obs_dim）
        # Bootstrapped uncertainty
        target_q_pred = self.target_critic(next_state_batch, next_action_batch)
        ucb_t = torch.std(target_q_pred, dim=0, keepdim=True)
        return ucb_t, target_q_pred
    def ood_critic_loss(self, state, action,next_state) :
        weight_of_ood_l2 = self.w_schedule.value(self.num_q_update_steps)
        state_temp = state.unsqueeze(1).repeat(1, self.random_num_samples, 1).view(
            state.shape[0] * self.random_num_samples, state.shape[1])
        state_temp_actions, _ = self.actor(state_temp, need_log_prob=True)
        next_state_temp = next_state.unsqueeze(1).repeat(1, self.random_num_samples, 1).view(
            next_state.shape[0] * self.random_num_samples, next_state.shape[1])
        next_state_temp_actions, _ = self.actor(next_state_temp, need_log_prob=True)
        # 计算ucb
        ucb_curr_actions, q_curr_actions_all = self.ucb_func(state, state_temp_actions)
        ucb_next_actions, q_next_actions_all = self.ucb_func(next_state, next_state_temp_actions)
        assert ucb_curr_actions.size() == ucb_next_actions.size()
        cat_q_ood = torch.cat([q_curr_actions_all, q_next_actions_all], 1)
        if self.vae_mask:
            with torch.no_grad():
                ood_vae_loss = self.vae.elbo_loss(state_temp, state_temp_actions, 0.5, 1).unsqueeze(0).clip(self.vae_loss_clip_min,self.vae_loss_clip_max)
                ood_vae_loss_mean = ood_vae_loss.mean().item()
                ood_vae_loss_min = ood_vae_loss.min().item()
                ood_vae_loss_max = ood_vae_loss.max().item()
                ood_vae_next_loss = self.vae.elbo_loss(next_state_temp, next_state_temp_actions, 0.5, 1).unsqueeze(0).clip(self.vae_loss_clip_min,self.vae_loss_clip_max)
            a = q_curr_actions_all - weight_of_ood_l2 * ood_vae_loss * ucb_curr_actions  # 2560*1
            b = q_next_actions_all - 0.1 * ood_vae_next_loss* ucb_next_actions
        else:
            ood_vae_loss_mean = 0.0
            ood_vae_loss_min = 0.0
            ood_vae_loss_max = 0.0
            a = q_curr_actions_all - weight_of_ood_l2 * ucb_curr_actions  # 2560*1
            b = q_next_actions_all - 0.1 * ucb_next_actions
        cat_q_ood_target = torch.cat([
            torch.maximum(a, torch.zeros(1).to(self.device)),
            torch.maximum(b, torch.zeros(1).to(self.device))], 1).detach()
        ood_critic_loss = (self.qf_criterion(cat_q_ood, cat_q_ood_target))
        ood_critic_loss = ood_critic_loss.mean()
        if self.eval_log :
            with torch.no_grad():
                batch_size = state.shape[0]
                action_random = torch.distributions.uniform.Uniform(-1, 1).sample((batch_size, self.action_dim))
                action_noise1 =  action + torch.normal(mean=torch.zeros((batch_size, self.action_dim)), std=torch.ones((batch_size, self.action_dim)) * 0.1)
                action_noise2 =  action + torch.normal(mean=torch.zeros((batch_size, self.action_dim)), std=torch.ones((batch_size, self.action_dim)) * 0.5)
                action_noise3 =  action + torch.normal(mean=torch.zeros((batch_size, self.action_dim)), std=torch.ones((batch_size, self.action_dim)) * 1.0)
                ood_rand = self.vae.elbo_loss(state, action_random, 0.5, 1).unsqueeze(0)
                ood_noise1 = self.vae.elbo_loss(state, action_noise1, 0.5, 1).unsqueeze(0)
                ood_noise2 = self.vae.elbo_loss(state, action_noise2, 0.5, 1).unsqueeze(0)
                ood_noise3 = self.vae.elbo_loss(state, action_noise3, 0.5, 1).unsqueeze(0)
                ood_real = self.vae.elbo_loss(state, action, 0.5, 1).unsqueeze(0)
                return ood_critic_loss, {
                    "q_cur_policy": q_curr_actions_all.mean().item(),
                    "ood_rand_loss_mean":ood_rand.mean().item(),
                    "ood_rand_loss_min":ood_rand.min().item(),
                    "ood_rand_loss_max":ood_rand.max().item(),
                    "ood_noise1_loss_mean": ood_noise1.mean().item(),
                    "ood_noise1_loss_min": ood_noise1.min().item(),
                    "ood_noise1_loss_max": ood_noise1.max().item(),
                    "ood_noise2_loss_mean": ood_noise2.mean().item(),
                    "ood_noise2_loss_min": ood_noise2.min().item(),
                    "ood_noise2_loss_max": ood_noise2.max().item(),
                    "ood_noise3_loss_mean": ood_noise3.mean().item(),
                    "ood_noise3_loss_min": ood_noise3.min().item(),
                    "ood_noise3_loss_max": ood_noise3.max().item(),
                    "ood_real_loss_mean": ood_real.mean().item(),
                    "ood_real_loss_min": ood_real.min().item(),
                    "ood_real_loss_max": ood_real.max().item(),
                    "ood_vae_loss_mean": ood_vae_loss_mean,
                    "ood_vae_loss_min": ood_vae_loss_min,
                    "ood_vae_loss_max": ood_vae_loss_max,
                }

        return ood_critic_loss,{
            "ood_vae_loss_mean":ood_vae_loss_mean,
            "ood_vae_loss_min": ood_vae_loss_min,
            "ood_vae_loss_max": ood_vae_loss_max,
            "q_cur_policy": q_curr_actions_all.mean().item()}

    def critic_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        done: torch.Tensor,
    ) :
        self.num_q_update_steps +=1
        with torch.no_grad():
            next_action, next_action_log_prob = self.actor(next_state, need_log_prob=True)
            # q_next = self.target_critic(next_state, next_action).min(0).values
            ucb_next, q_next = self.ucb_func_target(next_state, next_action)
            if self.q_mode == 'min':
                q_next = q_next.min(0).values
            elif self.q_mode == 'ave':
                q_next = q_next.mean(0)
            elif self.q_mode == 'rem':
                random_idx = np.random.permutation(self.num_critics)  # 随机选择critic网络
                q_next = q_next[random_idx]
                q_next1, q_next2 = q_next[:2]
                q_next = torch.min(q_next1, q_next2)
            if self.backup_entropy :
                q_next = q_next - self.alpha * next_action_log_prob
            ucb_next = ucb_next.transpose(0,1)
            assert q_next.unsqueeze(-1).shape == done.shape == reward.shape
            if self.vae_mask:
                next_state_action_vae_loss = self.vae.elbo_loss(next_state, next_action, 0.5, 1).unsqueeze(1).clip(self.vae_loss_clip_min,self.vae_loss_clip_max)
                q_target = reward + self.gamma * (1 - done) * (q_next.unsqueeze(-1)- self.ucb_ratio * next_state_action_vae_loss * ucb_next)
            else:
                q_target = reward + self.gamma * (1 - done) * (q_next.unsqueeze(-1) - self.ucb_ratio * ucb_next)
        q_values = self.critic(state, action)
        y = q_target.view(1, -1)
        # [ensemble_size, batch_size] - [1, batch_size]
        critic_loss = ((q_values - y) ** 2).mean(dim=1).sum(dim=0)
        # diversity_loss = self._critic_diversity_loss(state, action)
        ood_loss,ood_loss_logs = self.ood_critic_loss(state, action, next_state)
        loss = critic_loss + ood_loss
        return loss,{**ood_loss_logs}
    def learn(self, batch_list):
        state = torch.Tensor(np.array(batch_list["state_list"])).to(self.device)
        next_state = torch.Tensor(np.array(batch_list["next_state_list"])).to(self.device)
        action = torch.Tensor(np.array(batch_list["action_list"])).to(self.device)
        reward = torch.Tensor(np.array(batch_list["reward_list"])).to(self.device).unsqueeze(1)
        done = torch.Tensor(np.array(batch_list["done_list"])).to(self.device).unsqueeze(1)
        # Usually updates are done in the following order: critic -> actor -> alpha
        # But we found that EDAC paper uses reverse (which gives better results)
        # Alpha update
        alpha_loss = self.alpha_loss(state)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp().detach()
        # Actor update
        actor_loss, actor_logs = self.actor_loss(state)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Critic update
        critic_loss,critic_loss_logs = self.critic_loss(state, action, reward, next_state, done)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        #  Target networks soft update
        with torch.no_grad():
            soft_update(self.target_critic, self.critic, tau=self.tau)
            # for logging, Q-ensemble std estimate with the random actions:
            # a ~ U[-max_action, max_action]
            max_action = self.actor.max_action
            random_actions = -max_action + 2 * max_action * torch.rand_like(action)
            q_random_std = self.critic(state, random_actions).std(0).mean().item()
        update_info = {
            # "alpha_loss": alpha_loss.item(),
            # "critic_loss": critic_loss.item(),
            # "actor_loss": actor_loss.item(),
            # "alpha": self.alpha.item(),
            "q_random_std": q_random_std,
            **critic_loss_logs}
        return update_info
