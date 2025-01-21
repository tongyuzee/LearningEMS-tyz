import torch
import collections 
import random
import argparse
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from env.PriusV0 import PriusEnv
from env.RISSatComEnv_v1 import RISSatComEnv
import json
from datetime import datetime
import time
import matplotlib.pyplot as plt


PATH1 = "./Models/SAC/WLTC_"
PATH2 = "./Result/SAC/WLTC_"

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
    plt.grid(True)
    plt.savefig(file_name)
    # plt.show()
    plt.close()

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_args():
    """ 
        超参数
    """
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name', default='SAC', type=str, help="name of algorithm")
    parser.add_argument('--train_eps', default=10000, type=int, help="episodes of training")
    parser.add_argument('--test_eps', default=20, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.9, type=float, help="discounted factor")
    parser.add_argument('--actor_lr', default=1e-4/2, type=float)
    parser.add_argument('--critic_lr', default=1e-3/2, type=float)
    parser.add_argument('--alpha_lr', default=1e-3/2, type=float)
    parser.add_argument('--memory_capacity', default=100000, type=int, help="memory capacity")
    parser.add_argument('--minimal_size', default=1000, type=int, help="memory capacity")
    parser.add_argument('--batch_size', default=1024, type=int)
    parser.add_argument('--soft_tau', default=0.005, type=float)
    parser.add_argument('--hidden_dim', default=512, type=int)
    parser.add_argument('--hidden_dim1', default=256, type=int)
    parser.add_argument('--seed', default=516, type=int, help="random seed")

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
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(PolicyNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound

        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.zeros_(self.fc1.bias)

    def forward(self, x):
        if torch.isnan(x).any():
            raise ValueError("Input contains NaN or Inf values")
        x1 = F.relu(self.fc1(x))
        if torch.isnan(x1).any():
            raise ValueError("x1 contains NaN or Inf values")
        mu = self.fc_mu(x1)
        std = F.softplus(self.fc_std(x1)) + 1e-6
        dist = Normal(mu, std)
        normal_sample = dist.rsample()  

        # action = torch.tanh(normal_sample)
        # action = action * self.action_bound
        action = torch.sigmoid(normal_sample)  # 限制在[0,1]之间
        action = torch.clamp(action, 1e-6, 1)  # 限制在[0,1]之间

        log_prob = dist.log_prob(normal_sample)
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdims=True) # 计算熵 是否需要加和？
        action = action * self.action_bound
        return action, log_prob


class QValueNetContinuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, hidden_dim1, action_dim):
        super(QValueNetContinuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim1)
        self.fc_out = torch.nn.Linear(hidden_dim1, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        x = F.relu(self.fc1(cat))
        x = F.relu(self.fc2(x))
        return self.fc_out(x)


class ReplayBuffer(object):
    def __init__(self, state_dim, action_dim):
        self.max_size = int(100000)
        self.count = 0
        self.size = 0
        self.s = np.zeros((self.max_size, state_dim))
        self.a = np.zeros((self.max_size, action_dim))
        self.r = np.zeros((self.max_size, 1))
        self.s_ = np.zeros((self.max_size, state_dim))
        self.dw = np.zeros((self.max_size, 1))

    def store(self, s, a, r, s_, dw):
        self.s[self.count] = s
        self.a[self.count] = a
        self.r[self.count] = r
        self.s_[self.count] = s_
        self.dw[self.count] = dw
        self.count = (self.count + 1) % self.max_size 
        self.size = min(self.size + 1, self.max_size) 

    def sample(self,batch_size):
        index = np.random.choice(self.size, size=batch_size)
        batch_s = torch.tensor(self.s[index], dtype=torch.float)
        batch_a = torch.tensor(self.a[index], dtype=torch.float)
        batch_r = torch.tensor(self.r[index], dtype=torch.float)
        batch_s_ = torch.tensor(self.s_[index], dtype=torch.float)
        batch_dw = torch.tensor(self.dw[index], dtype=torch.float)

        return batch_s, batch_a, batch_r, batch_s_, batch_dw


class SACContinuous:
    def __init__(self, state_dim, action_dim, hidden_dim, hidden_dim1, action_bound, target_entropy, cfg):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, action_dim, action_bound).to(device)  
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim, hidden_dim1, action_dim).to(device)  
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim, hidden_dim1, action_dim).to(device)  
        self.target_critic_1 = QValueNetContinuous(state_dim, hidden_dim, hidden_dim1, action_dim).to(device)  
        self.target_critic_2 = QValueNetContinuous(state_dim,hidden_dim, hidden_dim1, action_dim).to(device)  
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),lr = cfg['actor_lr'])
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),lr = cfg['critic_lr'])
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),lr = cfg['critic_lr'])

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True 
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],lr = cfg['alpha_lr'])

        self.actor_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.actor_optimizer, milestones=[95], gamma=0.1)
        self.critic_1_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.critic_1_optimizer, milestones=[95], gamma=0.1)
        self.critic_2_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.critic_2_optimizer, milestones=[95], gamma=0.1)
        self.log_alpha_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.log_alpha_optimizer, milestones=[95], gamma=0.1)


        self.writer = SummaryWriter("Logs_WLTC/SAC_HEV0")

        self.target_entropy = target_entropy
        self.gamma = cfg['gamma']
        self.tau = cfg['soft_tau']
        self.batch_size = cfg['batch_size']
        self.memory = ReplayBuffer(state_dim, action_dim)

        self.M = cfg['num_RIS_elements']
        self.N = cfg['num_antennas']
        self.T = cfg['num_users']
        self.I = cfg['num_satellite']
        self.power_t = cfg['power_t']

    def compute_abs(self, a):
        # 求w的欧几里得范数，||w||
        w_real = a[: self.N * self.I].detach()
        w_imag = a[self.N * self.I:2 * self.N * self.I].detach()
        w = w_real.reshape(self.I, self.N) + 1j * w_imag.reshape(self.I, self.N)
        return torch.norm(w, dim=1, keepdim=True).expand(-1,self.N)
    
    # def compute_phase(self, a):
    #     # 规范化相位矩阵
    #     Phi_real = a[-2 * self.M * self.T:-self.M * self.T].detach()
    #     Phi_imag = a[-self.M * self.T:].detach()
    #     phi = torch.abs(Phi_real + 1j * Phi_imag)
    #     return phi

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(device)
        action = self.actor(state)[0].detach().cpu()
        
        action = torch.sigmoid(action)  # 限制在[0,1]之间
        action = torch.clamp(action, 1e-6, 1)  # 限制在[0,1]之间

        wabs = self.compute_abs(action.detach()).reshape(-1)
        action[: 2 * self.N * self.I] = action[: 2 * self.N * self.I] / wabs.repeat(2)
        # action = action.clamp(-1, 1)
        return action.cpu().numpy().reshape(-1)

    def calc_target(self, rewards, next_states, dones):  # 计算目标Q值
        next_actions, log_prob = self.actor(next_states)  # 计算下一步的动作和熵entropy
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value,q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)
        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def update(self):
        if self.memory.size < self.batch_size: 
            return
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        states = state.clone().detach().to(device)
        actions = action.clone().detach().to(device)
        rewards = reward.clone().detach().view(-1, 1).to(device)
        next_states = next_state.clone().detach().to(device)
        dones = done.clone().detach().view(-1, 1).to(device)
        
        # 计算目标 Q 值（Critic 更新）
        td_target = self.calc_target(rewards, next_states, dones)
        critic_1_loss = torch.mean(F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(F.mse_loss(self.critic_2(states, actions), td_target.detach()))
        # 更新 Critic 网络
        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 从 Actor 中获取新的动作和 log_prob
        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)
        actor_loss = torch.mean(-self.log_alpha.exp() * entropy - torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean((entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        # # 学习率调整
        # self.actor_scheduler.step()
        # self.critic_1_scheduler.step()
        # self.critic_2_scheduler.step()
        # self.log_alpha_scheduler.step()

    def save(self, current_time):
        torch.save(self.actor.state_dict(), PATH1 + f"actor_parameters_{current_time}.path")
        torch.save(self.critic_1.state_dict(), PATH1 + f"critic1_parameters_{current_time}.path")
        torch.save(self.critic_2.state_dict(), PATH1 + f"critic2_parameters_{current_time}.path")
        print("====================================")
        print("Model has been saved!!!")
        print("====================================")

    def deal(self, list0, list1, list2, list3):
        df0 = pd.DataFrame(list0, columns=['Reward'])
        df1 = pd.DataFrame(list1, columns=['Cost'])
        df2 = pd.DataFrame(list2, columns=['SOC'])
        df3 = pd.DataFrame(list3, columns=['SOC_Last'])
        df0.to_excel(PATH2 + "Reward.xlsx", index=False)
        df1.to_excel(PATH2 + "Cost.xlsx", index=False)
        df2.to_excel(PATH2 + "SOC.xlsx", index=False)
        df3.to_excel(PATH2 + "SOC_Last.xlsx", index=False)


def main():
    cfg = get_args()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{cfg['num_antennas']}_{cfg['num_RIS_elements']}_{cfg['num_satellite']}_{cfg['power_t']}_{cfg['gamma']}_{cfg['actor_lr']:1.0e}_{cfg['critic_lr']:1.0e}_{cfg['alpha_lr']:1.0e}_seed{cfg['seed']:05d}_{current_time}"
    with open(f"./Learning_Curves/{cfg['algo_name']}/{file_name}.txt", 'w') as f:
        json.dump(cfg, f, indent=4)
    f.close()

    set_seed(cfg['seed'])

    env = RISSatComEnv(cfg['num_antennas'], cfg['num_RIS_elements'], cfg['num_users'], cfg['num_satellite'], cfg['seed'], power_t=cfg['power_t'], channel_est_error=cfg['channel_est_error'])
    state_dim = env.state_dim
    action_dim = env.action_dim
    # action_bound = env.action_space.high[0]
    action_bound = 1
    target_entropy = -env.action_dim
    hidden_dim = cfg['hidden_dim']
    hidden_dim1 = cfg['hidden_dim1']
    agent = SACContinuous(state_dim, action_dim, hidden_dim, hidden_dim1, action_bound, target_entropy, cfg)

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
        while not done:

            if total_steps < cfg['test_eps']: 
                action = env.sample_action()
            else:
                action = agent.take_action(state)
            next_state, reward, done, info = env.step(action)

            agent.memory.store(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if reward > max_reward:
                max_reward = reward
            episode_steps += 1
            if total_steps >= cfg['test_eps']:
                agent.update()
        # 学习率调整
        # agent.actor_scheduler.step()
        # agent.critic_1_scheduler.step()
        # agent.critic_2_scheduler.step()
        # agent.log_alpha_scheduler.step()
       
        if episode_reward > max_eps_reward:
            max_eps_reward = episode_reward
            agent.save(current_time)
        Reward_list.append(episode_reward)  
        MaxReward_list.append(max_reward)
        eps_time = time.time()
        print(f"\nEpisode_Num:{total_steps:04d}   Episode_Steps:{episode_steps}    Episode_Time:{eps_time-start_time:07.1f}s    Max_resawr:{max_reward:.3f}    Episode_Reward:{episode_reward:.3f}\n")
        if (total_steps + 1)  % 100 == 0 :
            np.save(f"./Learning_Curves/{cfg['algo_name']}/{file_name}_eps_{total_steps:04d}", Reward_list)
            plot_learning_curves(Reward_list, MaxReward_list, f"./Learning_Curves/{cfg['algo_name']}/{file_name}_eps_{total_steps:04d}.png")
        
    plot_learning_curves(Reward_list, MaxReward_list, f"./Learning_Curves/{cfg['algo_name']}/{file_name}.png")
    agent.save(current_time + '_end')

def compare():
    """SAC与AO算法效果比较"""
    cfg = get_args()
    # current_time = '20250116_183957'
    current_time = '20250121_001638'
    file_name = f"{cfg['num_antennas']}_{cfg['num_RIS_elements']}_{cfg['num_satellite']}_{cfg['power_t']}_{cfg['gamma']}_{cfg['actor_lr']:1.0e}_{cfg['critic_lr']:1.0e}_{cfg['alpha_lr']:1.0e}_seed{cfg['seed']:05d}_{current_time}"

    env = RISSatComEnv(cfg['num_antennas'], cfg['num_RIS_elements'], cfg['num_users'], cfg['num_satellite'], cfg['seed'], power_t=cfg['power_t'], channel_est_error=cfg['channel_est_error'])
    state_dim = env.state_dim
    action_dim = env.action_dim
    # action_bound = env.action_space.high[0]
    action_bound = 1
    target_entropy = -env.action_dim
    hidden_dim = cfg['hidden_dim']
    hidden_dim1 = cfg['hidden_dim1']
    agent = SACContinuous(state_dim, action_dim, hidden_dim, hidden_dim1, action_bound, target_entropy, cfg)
    # 加载已有模型
    agent.actor.load_state_dict(torch.load(PATH1 + f"actor_parameters_{current_time}.path", map_location=device, weights_only=True))
    agent.critic_1.load_state_dict(torch.load(PATH1 + f"critic1_parameters_{current_time}.path", map_location=device, weights_only=True))
    agent.critic_2.load_state_dict(torch.load(PATH1 + f"critic2_parameters_{current_time}.path", map_location=device, weights_only=True))

    state = env.reset()
    done = False
    AOLr = []
    SACr = []
    episode_steps = 0
    while not done:
        AOreward, _, _ = env.AO_Low(env.h, env.H, env.g, env.sigema)
        AOLr.append(AOreward-60)
        action = agent.take_action(state)
        next_state, reward, done, info = env.step(action)
        SACr.append(reward-60)
        state = next_state
        episode_steps += 1
    plt.figure(figsize=(10, 6))
    plt.plot(AOLr, label='AO Rewards')
    plt.plot(SACr, label='SAC Reward')
    plt.legend()  # 显示图例
    plt.xlabel('time')
    plt.ylabel('Reward')
    plt.title(f"N={cfg['num_antennas']}, M={cfg['num_RIS_elements']}, I={cfg['num_satellite']}")
    plt.grid(True)
    plt.show(block=False)
    plt.savefig(f"./Learning_Curves/{cfg['algo_name']}/{file_name}_compare.png")

if __name__ == '__main__':
    # main()
    compare()