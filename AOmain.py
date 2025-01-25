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


def get_args():
    """ 
        超参数
    """
    parser = argparse.ArgumentParser(description="hyperparameters")      
    parser.add_argument('--algo_name', default='AO', type=str, help="name of algorithm")
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

    parser.add_argument("--num_antennas", default=16, type=int, metavar='N', help='Number of antennas in per satellite')
    parser.add_argument("--num_RIS_elements", default=1600, type=int, metavar='N', help='Number of RIS elements')
    parser.add_argument("--num_users", default=1, type=int, metavar='N', help='Number of users')
    parser.add_argument("--num_satellite", default=3, type=int, metavar='N', help='Number of satellite')
    parser.add_argument("--power_t", default=120, type=float, metavar='N', help='Transmission power for the constrained optimization in dB')
    parser.add_argument("--awgn_var", default=1e-2, type=float, metavar='G', help='Variance of the additive white Gaussian noise (default: 0.01)')
    parser.add_argument("--channel_est_error", default=False, type=bool, help='Noisy channel estimate? (default: False)')

    args = parser.parse_args([])    
    args = {**vars(args)}          
    return args

def main():
    # env = PriusEnv()
    cfg = get_args()
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{cfg['num_antennas']}_{cfg['num_RIS_elements']}_{cfg['num_satellite']}_{cfg['power_t']}_{cfg['gamma']}_{cfg['lmbda']}_{cfg['actor_lr']:1.0e}_{cfg['critic_lr']:1.0e}_seed{cfg['seed']:05d}_{current_time}"
    with open(f"./Learning_Curves/{cfg['algo_name']}/{file_name}.txt", 'w') as f:
        json.dump(cfg, f, indent=4)
    f.close()

    if cfg['seed'] is not None:
        torch.manual_seed(cfg['seed'])
        np.random.seed(cfg['seed'])

    if cfg['LOAD_MODEL']:

        AOr_RIS = []
        AOr_RIS0 = []
        env = RISSatComEnv(cfg['num_antennas'], cfg['num_RIS_elements'], cfg['num_users'], cfg['num_satellite'],cfg['seed'], AWGN_var=cfg['awgn_var'], power_t=cfg['power_t'], channel_est_error=cfg['channel_est_error'])
        state = env.reset()
        done = False
        episode_steps = 0
        while not done:
            AOreward1, _, _ = env.AO_Low(env.h, env.H, env.g, env.sigema)
            AOr_RIS.append(AOreward1-60)
            AO0rwared0, _, _ = env.AO0(env.h, env.H, env.g)
            AOr_RIS0.append(AO0rwared0-60)
            next_state, reward, done, info = env.step(None)
            episode_steps += 1

        """不部署RIS, cfg['num_RIS_elements']=0"""
        AOr_nRIS = []
        AOr_nRIS0 = []
        env = RISSatComEnv(cfg['num_antennas'], 0, cfg['num_users'], cfg['num_satellite'],cfg['seed'], AWGN_var=cfg['awgn_var'], power_t=cfg['power_t'], channel_est_error=cfg['channel_est_error'])
        state = env.reset()
        done = False
        episode_steps = 0
        while not done:
            AOreward1, _, _ = env.AO_Low(env.h, env.H, env.g, env.sigema)
            AOr_nRIS.append(AOreward1-60)
            AO0rwared0, _, _ = env.AO0(env.h, env.H, env.g)
            AOr_nRIS0.append(AO0rwared0-60)
            next_state, reward, done, info = env.step(None)
            episode_steps += 1    

        plt.figure(figsize=(10, 6))
        plt.plot(AOr_RIS, label='AO-RIS Low', linestyle='-', color='blue')  
        plt.plot(AOr_RIS0, label='AO-RIS Origin', linestyle='--', color='blue')  
        plt.plot(AOr_nRIS, label='AO-nRIS Low', linestyle='-', color='green')  
        plt.plot(AOr_nRIS0, label='AO-nRIS Origin', linestyle='--', color='green')  
        # plt.plot(np.array(AOr_RIS) - np.array(AOr_nRIS), label='error', linestyle='--')
        # plt.plot(DRLr, label='PPO Reward')
        # 显示图例
        plt.legend()
        plt.xlabel('time')
        plt.ylabel('Reward')
        plt.title(f"N={cfg['num_antennas']}, M={cfg['num_RIS_elements']}, I={cfg['num_satellite']}")
        plt.grid(True)
        plt.show(block=False)
        plt.savefig(f"./Learning_Curves/{cfg['algo_name']}/{file_name}.png")

if __name__ == '__main__':
    main()
 