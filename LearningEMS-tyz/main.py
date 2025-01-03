import torch
import numpy as np
import collections 
import random
import argparse
# from DDPG import DDPG
from PPO import PPOContinuous
from env.RISSatComEnv import RISSatComEnv
import time
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def size(self):
        return len(self.buffer)
    

def plot_learning_curves(rewards, file_name):
    """绘制学习曲线"""
    # path = os.path.join(file_name)
    plt.figure(figsize=(10, 6))
    plt.plot(rewards, label='Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Curve')
    plt.legend()
    plt.show()
    plt.savefig(file_name)

def get_args():
    """
        超参数
    """
    parser = argparse.ArgumentParser(description="hyperparameters")
    parser.add_argument('--algo_name', default='DDPG', type=str, help="name of algorithm")
    parser.add_argument('--train_eps', default=50000, type=int, help="episodes of training")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--lmbda', default=0.98, type=float)
    parser.add_argument('--critic_lr', default=2e-4, type=float, help="learning rate of critic")
    parser.add_argument('--actor_lr', default=1e-4, type=float, help="learning rate of actor")
    parser.add_argument('--memory_capacity', default=100000, type=int, help="memory capacity")
    parser.add_argument('--minimal_size', default=1000, type=int, help="memory capacity")
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--tau', default=0.005, type=float)
    parser.add_argument('--exploration_noise', default=0.1, type=float)
    parser.add_argument('--hidden_dim', default=200, type=int)
    parser.add_argument('--hidden_dim1', default=100, type=int)
    parser.add_argument('--seed', default=0, type=int, help="random seed")

    parser.add_argument("--num_antennas", default=4, type=int, metavar='N', help='Number of antennas in per satellite')
    parser.add_argument("--num_RIS_elements", default=16, type=int, metavar='N', help='Number of RIS elements')
    parser.add_argument("--num_users", default=1, type=int, metavar='N', help='Number of users')
    parser.add_argument("--num_satellite", default=3, type=int, metavar='N', help='Number of satellite')
    parser.add_argument("--power_t", default=120, type=float, metavar='N', help='Transmission power for the constrained optimization in dB')
    parser.add_argument("--awgn_var", default=1e-2, type=float, metavar='G', help='Variance of the additive white Gaussian noise (default: 0.01)')
    parser.add_argument("--channel_est_error", default=False, type=bool, help='Noisy channel estimate? (default: False)')
    args = parser.parse_args([])
    args = {**vars(args)}
    return args

if __name__ == "__main__":
    cfg = get_args()
    file_name = f"{cfg['num_antennas']}_{cfg['num_RIS_elements']}_{cfg['num_satellite']}_{cfg['power_t']}_{cfg['gamma']}"
    env = RISSatComEnv(cfg['num_antennas'], cfg['num_RIS_elements'], cfg['num_users'], cfg['num_satellite'], AWGN_var=cfg['awgn_var'], power_t=cfg['power_t'], channel_est_error=cfg['channel_est_error'])
    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])
    state_dim = env.state_dim
    action_dim = env.action_dim
    hidden_dim = cfg['hidden_dim']
    hidden_dim1 = cfg['hidden_dim1']
    replay_buffer = ReplayBuffer(cfg['memory_capacity'])

    # agent = PPOContinuous(state_dim, action_dim, hidden_dim, hidden_dim1, cfg)
    max_action = 1
    agent = PPOContinuous(state_dim, hidden_dim, hidden_dim1, action_dim, max_action, cfg)

    Reward_list = []
    start_time = time.time()
    for total_steps in range(cfg['train_eps']):

        episode_reward = 0
        state = env.reset()
        done = False
        episode_steps = 0
        max_reward = -1e9
        while not done:
            action = agent.select_action(state)
            action = action + np.random.normal(0, cfg['exploration_noise'], size=env.action_dim)
            next_state, reward, done, info = env.step(action)

            # if total_steps == cfg['train_eps'] - 2:
            #     SOC_last_list.append(float(info['SOC']))
            #     num_SOC_last += 1
            #     agent.writer.add_scalar('SOC_last', float(info['SOC']), global_step=num_SOC_last)

            replay_buffer.add(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            if reward > max_reward:
                max_reward = reward

            if replay_buffer.size() > cfg['minimal_size']:
                b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(cfg['batch_size'])
                transition_dict = {
                    'states': b_s,
                    'actions': b_a,
                    'next_states': b_ns,
                    'rewards': b_r,
                    'dones': b_d
                }
                agent.update(transition_dict)
            
            episode_steps += 1

        # agent.writer.add_scalar('Reward', episode_return, global_step=total_steps)
        # agent.writer.add_scalar('Cost', info['Total_cost'], global_step=total_steps)
        # agent.writer.add_scalar('SOC', info['SOC'], global_step=total_steps)
        # Cost_list.append(info['Total_cost'])
        # SOC_list.append(float(info['SOC']))
        Reward_list.append(episode_reward) 

        eps_time = time.time()
        print(f"\nEpisode_Num: {total_steps} Episode_Steps: {episode_steps} Episode_Time: {eps_time-start_time:.3f}s Max_resawr: {max_reward:.3f} Episode_Reward: {episode_reward:.3f}\n")


        # if total_steps == cfg['train_eps'] - 1:
        #     agent.deal(Reward_list, Cost_list, SOC_list, SOC_last_list)
        #     agent.save()

        # print("Cost:{} \nSOC :{} \nEpisode:{} \nTotal Reward: {:0.2f} \n".format(info['Total_cost'], float(info['SOC']), total_steps, episode_return))

    plot_learning_curves(Reward_list, f"./Learning_Curves/{agent.__class__.__name__ }/{file_name}.png")  