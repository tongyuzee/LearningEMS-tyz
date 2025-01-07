import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections 
import random
import argparse
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from env.PriusV0 import PriusEnv
from env.RISSatComEnv_v1 import RISSatComEnv
import time
import json
from datetime import datetime
import time
import matplotlib.pyplot as plt


PATH1 = "./Models/PPO/WLTC_"
PATH2 = "./Result/PPO/WLTC_"

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def plot_learning_curves(Rewards, MaxReward, file_name):
    """绘制学习曲线"""
    # path = os.path.join(file_name)
    plt.figure(figsize=(10, 6))
    plt.plot(Rewards, label='Rewards')
    # plt.plot(MaxReward, label='MaxReward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Curve')
    plt.legend()    
    plt.savefig(file_name)
    # plt.show()
    plt.close()

def get_args():
    """ 
        超参数
    """
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name', default='PPO', type=str, help="name of algorithm")
    parser.add_argument('--train_eps', default=8000, type=int, help="episodes of training")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--lmbda', default=0.999, type=float)
    parser.add_argument('--eps', default=0.2, type=float)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--actor_lr', default=1e-4/2, type=float)
    parser.add_argument('--critic_lr', default=1e-3/2, type=float)
    parser.add_argument('--hidden_dim', default=1024, type=int)
    parser.add_argument('--hidden_dim1', default=512, type=int)
    parser.add_argument('--seed', default=24, type=int, help="random seed")

    parser.add_argument('--LOAD_MODEL', default=True, type=bool, help="load model or not")

    parser.add_argument("--num_antennas", default=2, type=int, metavar='N', help='Number of antennas in per satellite')
    parser.add_argument("--num_RIS_elements", default=4, type=int, metavar='N', help='Number of RIS elements')
    parser.add_argument("--num_users", default=1, type=int, metavar='N', help='Number of users')
    parser.add_argument("--num_satellite", default=2, type=int, metavar='N', help='Number of satellite')
    parser.add_argument("--power_t", default=120, type=float, metavar='N', help='Transmission power for the constrained optimization in dB')
    parser.add_argument("--awgn_var", default=1e-2, type=float, metavar='G', help='Variance of the additive white Gaussian noise (default: 0.01)')
    parser.add_argument("--channel_est_error", default=False, type=bool, help='Noisy channel estimate? (default: False)')

    args = parser.parse_args([])    
    args = {**vars(args)}          
    return args


class PolicyNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, hidden_dim1, action_dim, max_action):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim1)
        self.fc_mu = torch.nn.Linear(hidden_dim1, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim1, action_dim)
        self.max_action = max_action

        # # 使用 Xavier 初始化
        # torch.nn.init.xavier_uniform_(self.fc1.weight)
        # torch.nn.init.xavier_uniform_(self.fc2.weight)
        # torch.nn.init.xavier_uniform_(self.fc_mu.weight)
        # torch.nn.init.xavier_uniform_(self.fc_std.weight)

        # # 初始化偏置
        # torch.nn.init.zeros_(self.fc1.bias)
        # torch.nn.init.zeros_(self.fc2.bias)
        # torch.nn.init.zeros_(self.fc_mu.bias)
        # torch.nn.init.zeros_(self.fc_std.bias)


    def forward(self, x):
        if torch.isnan(x).any():
            raise ValueError("Input contains NaN or Inf values")
        x1 = F.relu(self.fc1(x))
        if torch.isnan(x1).any():
            raise ValueError("NaN or Inf detected in fc1 output")
        x2 = F.relu(self.fc2(x1))
        if torch.isnan(x2).any():
            raise ValueError("NaN or Inf detected in fc2 output")
        mu = self.max_action * torch.tanh(self.fc_mu(x2))
        std = F.softplus(self.fc_std(x2)) 
        std = torch.max(std, torch.tensor(1e-6))  # 添加一个小的常数偏移量，防止标准差为0
        if torch.isnan(mu).any() or torch.isnan(std).any():
            raise ValueError("mu or sigma contains NaN values")
        return mu, std


class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, hidden_dim1):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim1)
        self.fc3 = torch.nn.Linear(hidden_dim1, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class PPOContinuous:
    def __init__(self, state_dim, hidden_dim, hidden_dim1, action_dim, max_action, cfg):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, hidden_dim1, action_dim, max_action).to(device)
        self.critic = ValueNet(state_dim, hidden_dim, hidden_dim1).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = cfg['actor_lr'])
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),lr = cfg['critic_lr'])

        self.gamma = cfg['gamma']
        self.lmbda = cfg['lmbda']
        self.epochs = cfg['epochs']
        self.eps = cfg['eps']
        self.writer = SummaryWriter("Logs_WLTC/PPO_HEV0")

        self.M = cfg['num_RIS_elements']
        self.N = cfg['num_antennas']
        self.T = cfg['num_users']
        self.I = cfg['num_satellite']
        self.power_t = cfg['power_t']

    def compute_power(self, a):
        # 求w的欧几里得范数，||w||
        w_real = a[: self.N * self.I].detach()
        w_imag = a[self.N * self.I:2 * self.N * self.I].detach()
        w = w_real.reshape(self.I, self.N) + 1j * w_imag.reshape(self.I, self.N)
        
        return torch.norm(w, dim=1, keepdim=True).expand(-1,self.N)

    # def compute_phase(self, a):
    #     # 规范化相位矩阵
    #     Phi_real = a[:, -2 * self.M * self.T:-self.M * self.T].detach()
    #     Phi_imag = a[:, -self.M * self.T:].detach()
    #     phi = torch.abs(Phi_real + 1j * Phi_imag)
    #     return phi

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(device)
        mu, sigma = self.actor(state)
        action_dist = torch.distributions.Normal(mu, sigma)
        action = action_dist.sample().reshape(-1)

        action = torch.sigmoid(action)  # 限制在[0,1]之间
        action = torch.clamp(action, 1e-6, 1)  # 限制在[0,1]之间

        # 规范化传输功率
        wabs = self.compute_power(action.detach()).reshape(-1)
        action[: 2 * self.N * self.I] = action[: 2 * self.N * self.I] / wabs.repeat(2)
        # normal = self.compute_phase(action.detach())
        # division_term = torch.cat([wabs, wabs, normal, normal], dim=1)
        # action  = action / division_term
        return action.cpu().numpy().reshape(-1)

    def update(self, transition_dict):
        states = torch.tensor(np.array(transition_dict['states']),dtype=torch.float).to(device)
        actions = torch.tensor(np.array(transition_dict['actions']),dtype=torch.float).to(device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1, 1).to(device)
        next_states = torch.tensor(np.array(transition_dict['next_states']),dtype=torch.float).to(device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1, 1).to(device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        advantage = compute_advantage(self.gamma, self.lmbda, td_delta.cpu()).to(device)
        mu, std = self.actor(states)
        action_dists = torch.distributions.Normal(mu.detach(), std.detach())
        old_log_probs = action_dists.log_prob(actions)

        for _ in range(self.epochs):
            # states = (states - states.mean(dim=0)) / (states.std(dim=0) + 1e-8)  # 归一化
            mu, std = self.actor(states)
            action_dists = torch.distributions.Normal(mu, std)
            log_probs = action_dists.log_prob(actions)
            # ratio = torch.exp(log_probs - old_log_probs)
            ratio = torch.exp(torch.clamp(log_probs - old_log_probs, -30, 30))
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))

            if torch.isnan(actor_loss).any() or torch.isinf(actor_loss).any():
                raise ValueError("NaN or Inf detected in actor_loss")

            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)  # 梯度裁剪
            critic_loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)  # 梯度裁剪
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def save(self, current_time):
        torch.save(self.actor.state_dict(), PATH1 + f"actor_parameters_{current_time}.path")
        torch.save(self.critic.state_dict(), PATH1 + f"critic_parameters_{current_time}.path")
        print("====================================")
        print("Model has been saved!!!")
        print("====================================")

    def load(self, current_time):
        self.actor.load_state_dict(torch.load(PATH1 + f"actor_parameters_{current_time}.path", map_location=device, weights_only=True))
        self.critic.load_state_dict(torch.load(PATH1 + f"critic_parameters_{current_time}.path", map_location=device, weights_only=True))

    def deal(self, list0, list1, list2, list3):
        df0 = pd.DataFrame(list0, columns=['Reward'])
        df1 = pd.DataFrame(list1, columns=['Cost'])
        df2 = pd.DataFrame(list2, columns=['SOC'])
        df3 = pd.DataFrame(list3, columns=['SOC_Last'])
        df0.to_excel(PATH2 + "Reward.xlsx", index=False)
        df1.to_excel(PATH2 + "Cost.xlsx", index=False)
        df2.to_excel(PATH2 + "SOC.xlsx", index=False)
        df3.to_excel(PATH2 + "SOC_Last.xlsx", index=False)

def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(np.array(advantage_list), dtype=torch.float)


def main():
    # env = PriusEnv()
    cfg = get_args()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{cfg['num_antennas']}_{cfg['num_RIS_elements']}_{cfg['num_satellite']}_{cfg['power_t']}_{cfg['gamma']}_{cfg['lmbda']}_{cfg['actor_lr']:1.0e}_{cfg['critic_lr']:1.0e}_seed{cfg['seed']:05d}_{current_time}"
    # with open(f"./Learning_Curves/{cfg['algo_name']}/{file_name}.txt", 'w') as f:
    #     json.dump(cfg, f, indent=4)
    # f.close()

    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
        
    env = RISSatComEnv(cfg['num_antennas'], cfg['num_RIS_elements'], cfg['num_users'], cfg['num_satellite'],cfg['seed'], AWGN_var=cfg['awgn_var'], power_t=cfg['power_t'], channel_est_error=cfg['channel_est_error'])
    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1
    hidden_dim = cfg['hidden_dim']
    hidden_dim1 = cfg['hidden_dim1']

    agent = PPOContinuous(state_dim, hidden_dim, hidden_dim1, action_dim, max_action, cfg)

    if cfg['LOAD_MODEL']:
        current_time = '20241219_095228'
        file_name = f"{cfg['num_antennas']}_{cfg['num_RIS_elements']}_{cfg['num_satellite']}_{cfg['power_t']}_{cfg['gamma']}_{cfg['lmbda']}_{cfg['actor_lr']:1.0e}_{cfg['critic_lr']:1.0e}_seed{cfg['seed']:05d}_{current_time}"
        
        agent.load(current_time)
        state = env.reset()
        done = False
        AOr = []
        AOr0 = []
        DRLr = []
        episode_steps = 0
        max_reward = -1e9
        while not done:
            AOreward, _, _ = env.AO_Low(env.h, env.H, env.g, env.sigema)
            AOr.append(AOreward)
            AO0rwared, _, _ = env.AO0(env.h, env.H, env.g)
            AOr0.append(AO0rwared)
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)
            DRLr.append(reward)
            state = next_state
            episode_steps += 1
            # episode_reward += reward
        plt.figure(figsize=(10, 6))
        plt.plot(AOr, label='AO Rewards')
        plt.plot(AOr0, label='AO0 Rewards')
        plt.plot(DRLr, label='PPO Reward')
        # 显示图例
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('Reward')
        plt.title('Reward Curve')
        plt.show(block=False)
        plt.savefig(f"./Learning_Curves/{cfg['algo_name']}/{file_name}_compare.png")
        

    Reward_list = []
    MaxReward_list = []
    max_eps_reward = -1e9
    start_time = time.time()
    for total_steps in range(cfg['train_eps']):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_steps = 0
        max_reward = -1e9
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        while not done:
            action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)

            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_steps += 1
            episode_reward += reward
            if reward > max_reward:
                max_reward = reward
            
        if episode_reward > max_eps_reward:
            max_eps_reward = episode_reward
            agent.save(current_time)

        agent.update(transition_dict)

        # agent.writer.add_scalar('Reward', episode_reward, global_step = total_steps)
        # agent.writer.add_scalar('Cost', info['Total_cost'], global_step = total_steps)
        # agent.writer.add_scalar('SOC', info['SOC'], global_step = total_steps)
        # Cost_list.append(info['Total_cost'])
        # SOC_list.append(info['SOC'])
        Reward_list.append(episode_reward)
        MaxReward_list.append(max_reward)
        # if total_steps == cfg['train_eps'] - 1:
        #     agent.deal(Reward_list, Cost_list, SOC_list, SOC_last_list)
        #     agent.save()

        eps_time = time.time()
        print(f"\nEpisode_Num:{total_steps:04d}   Episode_Steps:{episode_steps}    Episode_Time:{eps_time-start_time:07.1f}s    Max_Reward:{max_reward:.3f}    Episode_Reward:{episode_reward:.3f}\n")
        if (total_steps + 1)  % 200 == 0 :
        # if (total_steps + 1) >2:
            np.save(f"./Learning_Curves/{cfg['algo_name']}/{file_name}_eps_{total_steps:04d}", Reward_list)
            plot_learning_curves(Reward_list, MaxReward_list, f"./Learning_Curves/{cfg['algo_name']}/{file_name}_eps_{total_steps:04d}.png")

    np.save(f"./Learning_Curves/{cfg['algo_name']}/{file_name}", Reward_list)
    plot_learning_curves(Reward_list, MaxReward_list, f"./Learning_Curves/{cfg['algo_name']}/{file_name}.png")
    # agent.save(current_time)

if __name__ == '__main__':
    main()