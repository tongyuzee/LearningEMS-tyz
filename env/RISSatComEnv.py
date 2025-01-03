import numpy as np
import env.RISSatCom as RISSatCom


class RISSatComEnv:
    def __init__(self, 
                 num_antennas: int, 
                 num_RIS_elements: int, 
                 num_users: int,
                 num_satellites: int,
                 seed: int = 0,
                 channel_est_error: bool = False, 
                 AWGN_var: float = 1e-2,
                 channel_noise_var: float = 1e-2,
                 power_t: int = 120):
        
        self.T = num_users          # TR的天线数量
        self.N = num_antennas       # 卫星的天线数量
        self.M = num_RIS_elements   # RIS的元素数量
        self.I = num_satellites     # 卫星的数量
        
        self.power_t = power_t
        self.power_r = 0

        self.channel_est_error = channel_est_error
        self.awgn_var = AWGN_var
        self.channel_noise_var = channel_noise_var

        sat_comm = RISSatCom.RISSatCom(0, T=self.T, N=self.N, M=self.M, I=self.I)
        self.h, self.H, self.g = sat_comm.setup_channel()

        self.w = self.generate_unit_complex_numbers(self.I, self.N)
        self.Phi = self.generate_unit_complex_numbers(self.T, self.M)

        self.action_space = np.hstack((
            np.real(self.w).reshape(1, -1),
            np.imag(self.w).reshape(1, -1),
            np.real(self.Phi).reshape(1, -1),
            np.imag(self.Phi).reshape(1, -1)
        ))
        self.action_space = self.action_space[0]
        self.state_space = np.hstack((
            self.action_space.reshape(1, -1),
            np.real(self.h).reshape(1, -1),
            np.imag(self.h).reshape(1, -1),
            np.real(self.H).reshape(1, -1),
            np.imag(self.H).reshape(1, -1),
            # np.real(self.g).reshape(1, -1),
            # np.imag(self.g).reshape(1, -1),
            # np.array([[self.power_t, self.power_r]])
            np.array([[self.power_r]])
        ))
        self.state_space = self.state_space[0]

        self.action_dim = self.action_space.size
        self.state_dim = self.state_space.size
        self.done: bool = False
        self.episode_t: int = 0

        self.reward = 0
        self.epsilon = 1e-9
        self.seed = seed

    def reset(self) -> np.ndarray:
        """重置环境以开始新的一轮"""
        np.random.seed(self.seed)
        self.done: bool = False
        self.episode_t: int = 0
        self.reward = 0
        self.epsilon = 1e-9
        sat_comm = RISSatCom.RISSatCom(self.episode_t, T=self.T, N=self.N, M=self.M, I=self.I)
        self.h, self.H, self.g = sat_comm.setup_channel()
        self.w = self.generate_unit_complex_numbers(self.I, self.N)
        self.Phi = self.generate_unit_complex_numbers(self.T, self.M)

        self.action_space = np.hstack((
            np.real(self.w).reshape(1, -1),
            np.imag(self.w).reshape(1, -1),
            np.real(self.Phi).reshape(1, -1),
            np.imag(self.Phi).reshape(1, -1)
        ))
        self.action_space = self.action_space[0]
        self.state_space = np.hstack((
            self.action_space.reshape(1, -1),
            np.real(self.h).reshape(1, -1),
            np.imag(self.h).reshape(1, -1),
            np.real(self.H).reshape(1, -1),
            np.imag(self.H).reshape(1, -1),
            # np.real(self.g).reshape(1, -1),
            # np.imag(self.g).reshape(1, -1),
            # np.array([[self.power_t, self.power_r]])
            np.array([[self.power_r]])
        ))
        self.state_space = self.state_space[0]
        return self.state_space

    def step(self, action: np.ndarray) -> tuple:
        """执行动作并返回新的状态、奖励和结束标志"""
        sat_comm = RISSatCom.RISSatCom(self.episode_t , T=self.T, N=self.N, M=self.M, I=self.I)
        # sat_comm = RISSatCom.RISSatCom(100 , T=self.T, N=self.N, M=self.M, I=self.I)
        self.h, self.H, self.g = sat_comm.setup_channel()

        action = action.reshape(1, -1)
        w_real = action[:, :self.N * self.I]
        w_imag = action[:, self.N * self.I:2 * self.N * self.I]
        Phi_real = action[:, -2 * self.M * self.T:-self.M * self.T]
        Phi_imag = action[:, -self.M * self.T:]

        self.w = w_real.reshape(self.I, self.N) + 1j * w_imag.reshape(self.I, self.N)
        self.Phi = Phi_real.reshape(self.T, self.M) + 1j * Phi_imag.reshape(self.T, self.M)

        reward, _ = self._compute_reward()
        reward = reward 
        # 更新状态
        self.state_space = np.hstack((
            action.reshape(1, -1),
            np.real(self.h).reshape(1, -1),
            np.imag(self.h).reshape(1, -1),
            np.real(self.H).reshape(1, -1),
            np.imag(self.H).reshape(1, -1),
            # np.real(self.g).reshape(1, -1),
            # np.imag(self.g).reshape(1, -1),
            # np.array([[self.power_t, self.power_r]])
            np.array([[self.power_r]])
        ))
        self.state_space = self.state_space[0]

        done = self.episode_t >= sat_comm.TT
        info = {}
        if done:
            info = {
                'power_r': self.power_r,
                'power_t': self.power_t,
                'reward': reward
            }
        self.reward = reward
        self.episode_t += 1

        return self.state_space, reward, done, info

    def _compute_reward(self) -> tuple:
        """根据当前状态和动作计算奖励"""

        C = np.sum([np.abs(np.sum((self.h[i] + self.g * self.Phi @ self.H[i]) * self.w[i])) ** 2 for i in range(self.I)])
        self.power_r = self.power_t + 10 * np.log10(C)
        power_r = 10 ** (self.power_r / 10)
        # self.power_r = self.power_t * np.abs(np.sum((self.h + self.g * self.Phi @ self.H) * self.w)) ** 2
        # reward = 10 * np.log2(10 ** (self.power_r / 10))
        # reward = 10 * np.log10(C)
        reward = 10 * np.log2(power_r)
        opt_reward = 0  # 占位符，用于理想奖励的计算
        return reward, opt_reward

    def close(self):
        """清理环境资源"""
        pass

    def sample_action(self):
        """随机生成一个动作"""
        action = np.random.rand(self.action_dim)
        wabs = self.compute_power(action)
        phi = self.compute_phase(action)
        division_term = np.hstack([wabs, wabs, phi, phi])
        action  = action / division_term
        return action
    
    def compute_power(self, a):
        # 规范化功率
        w_real = a[:self.N * self.I]
        w_imag = a[self.N * self.I:2 * self.N * self.I]
        wabs = np.abs(w_real + 1j * w_imag)
        return wabs

    def compute_phase(self, a):
        # 规范化相位矩阵
        Phi_real = a[-2 * self.M * self.T:-self.M * self.T]
        Phi_imag = a[-self.M * self.T:]
        phi = np.abs(Phi_real + 1j * Phi_imag)
        return phi
    
    def generate_unit_complex_numbers(self, m, n):
        # 生成 n 个随机相位，范围在 [0, 2π)
        phases = np.random.rand(m, n) * 2 * np.pi
        # 使用欧拉公式生成复数
        complex_numbers = np.exp(1j * phases)
        return complex_numbers
