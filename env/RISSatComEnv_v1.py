import numpy as np
from gym import spaces
import env.RISSatCom as RISSatCom


def scale(x):
    # return x
    return 1e8 * x + 1


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
        self.RISactive = bool(self.M)  # 是否使用RIS
        
        self.power_t = power_t
        self.power_r = 0

        self.channel_est_error = channel_est_error
        self.awgn_var = AWGN_var
        self.channel_noise_var = channel_noise_var

        sat_comm = RISSatCom.RISSatCom(0, T=self.T, N=self.N, M=self.M, I=self.I)
        self.h, self.H, self.g, self.sigema = sat_comm.setup_channel()

        self.w = self.h / np.linalg.norm(self.h, axis=1, keepdims=True)
        phi = np.random.rand(self.T, self.M)
        self.Phi = np.exp(-1j * 2 * np.pi * phi)

        self.action_space = np.hstack((
            np.real(self.w).reshape(-1),
            np.imag(self.w).reshape(-1),
            phi.reshape(-1)
        ))        
        self.state_space = np.hstack((
            self.action_space.reshape(-1),
            np.real(scale(self.h)).reshape(-1),
            np.imag(scale(self.h)).reshape(-1),
            np.real(scale(self.H)).reshape(-1),
            np.imag(scale(self.H)).reshape(-1),
            # np.real(self.g).reshape(-1),
            # np.imag(self.g).reshape(-1),
            # np.array([self.power_t, self.power_r])
            np.array([self.power_r])
        ))

        self.action_dim = self.action_space.shape[0]
        self.state_dim = self.state_space.shape[0]
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
        self.h, self.H, self.g, self.sigema = sat_comm.setup_channel()

        self.w = self.h / np.linalg.norm(self.h, axis=1, keepdims=True)
        phi = np.random.rand(self.T, self.M)
        self.Phi = np.exp(-1j * 2 * np.pi * phi)

        self.action_space = np.hstack((
            np.real(self.w).reshape(-1),
            np.imag(self.w).reshape(-1),
            phi.reshape(-1)
        ))
        self.state_space = np.hstack((
            self.action_space.reshape(-1),
            np.real(scale(self.h)).reshape(-1),
            np.imag(scale(self.h)).reshape(-1),
            np.real(scale(self.H)).reshape(-1),
            np.imag(scale(self.H)).reshape(-1),
            # np.real(self.g).reshape(-1),
            # np.imag(self.g).reshape(-1),
            # np.array([self.power_t, self.power_r])
            np.array([self.power_r])
        ))
        return self.state_space

    def step(self, action: np.ndarray) -> tuple:
        """执行动作并返回新的状态、奖励和结束标志"""

        if action is None:
            """AO算法没有返回动作，仅需要当前的信道状态，信道状态与action无关"""
            action = self.sample_action()

        self.action_space = action

        w_real = action[:self.N * self.I]
        w_imag = action[self.N * self.I:2 * self.N * self.I]
        self.w = w_real.reshape(self.I, self.N) + 1j * w_imag.reshape(self.I, self.N)
        if self.M != 0:  # 不部署RIS, 无需计算Phi
            self.Phi = np.exp(-1j * 2 * np.pi * action[-self.T * self.M:].reshape(self.T, self.M))

        if (np.abs(np.linalg.norm(self.w, axis=1, keepdims=True) - 1) > 0.1).any():
            raise ValueError("The norm of w is not equal to 1!")

        reward, _ = self._compute_reward(self.h, self.H, self.g, self.w, self.Phi)
        sat_comm = RISSatCom.RISSatCom(self.episode_t , T=self.T, N=self.N, M=self.M, I=self.I)
        # sat_comm = RISSatCom.RISSatCom(100 , T=self.T, N=self.N, M=self.M, I=self.I)
        self.h, self.H, self.g, self.sigema = sat_comm.setup_channel()
        # 更新状态
        self.state_space = np.hstack((
            self.action_space.reshape(-1),
            np.real(scale(self.h)).reshape(-1),
            np.imag(scale(self.h)).reshape(-1),
            np.real(scale(self.H)).reshape(-1),
            np.imag(scale(self.H)).reshape(-1),
            # np.real(self.g).reshape(-1),
            # np.imag(self.g).reshape(-1),
            # np.array([self.power_t, self.power_r])
            np.array([self.power_r])
        ))
        done = self.episode_t >= sat_comm.TT
        info = {}
        if done:
            info = {
                'power_r': self.power_r,
                'power_t': self.power_t,
                'reward': reward
            }
        self.reward = reward
        self.episode_t += 10

        return self.state_space, reward, done, info

    def _compute_reward(self, h, H, g, w, Phi) -> tuple:
        """根据当前状态和动作计算奖励"""
        if self.RISactive:  # 使用RIS的信道容量
            C = np.abs(np.sum([np.sum((h[i] + g * Phi @ H[i]) * w[i]) for i in range(self.I)]))**2
            # C = np.sum([np.abs(np.sum((h[i] + g * Phi @ H[i]) * w[i])) ** 2 for i in range(self.I)])
        else:  # 不使用RIS的信道容量
            C = np.abs(np.sum([np.sum(h[i] * w[i]) for i in range(self.I)]))**2
            # C = np.sum([np.abs(np.sum(h[i] * w[i])) ** 2 for i in range(self.I)])
        # C = np.abs(np.sum([np.sum((h[i] + g * Phi @ H[i]) * w[i]) for i in range(self.I)]))**2
        # # C = np.sum([np.abs(np.sum((h[i] + g * Phi @ H[i]) * w[i])) ** 2 for i in range(self.I)])
        # C_nRIS = np.abs(np.sum([np.sum(h[i] * w[i]) for i in range(self.I)]))**2   # 不使用RIS的信道容量
        
        self.power_r = self.power_t + 10 * np.log10(C)
        power_r = 10 ** (self.power_r / 10)
        # self.power_r = self.power_t * np.abs(np.sum((self.h + self.g * self.Phi @ self.H) * self.w)) ** 2
        # reward = 10 * np.log2(10 ** (self.power_r / 10))
        # reward = 10 * np.log10(C)
        reward = 10 * np.log2(power_r)
        opt_reward = 0  # 占位符，用于理想奖励的计算
        reward = reward + 60  # 为了让奖励值有正有负，加上一个常数？
        return reward, opt_reward

    def close(self):
        """清理环境资源"""
        pass

    def sample_action(self):
        """随机生成一个动作"""
        action = np.random.rand(self.action_dim)
        # wabs = self.compute_power(action).repeat(self.N,axis=1).reshape(-1)
        # action[: 2 * self.N * self.I] = action[: 2 * self.N * self.I] / wabs.repeat(2)
        w = self.compute_power(action)
        action[: 2 * self.N * self.I] = np.hstack([np.real(w).reshape(-1), np.imag(w).reshape(-1)])
        # phi = self.compute_phase(action)
        # division_term = np.hstack([wabs, wabs, phi, phi])
        # action  = action / division_term
        return action
    
    def compute_power(self, a):
        # 求w的欧几里得范数，||w||，归一化
        w_real = a[: self.N * self.I]
        w_imag = a[self.N * self.I:2 * self.N * self.I]
        w = w_real.reshape(self.I, self.N) + 1j * w_imag.reshape(self.I, self.N)
        return w / np.linalg.norm(w, axis=1, keepdims=True)

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
    
    def compute_reward(self, h, H, g, sigma, w, phi):
        f_phi = 1 + np.sum(phi * sigma, axis=1, keepdims=True)
        T = np.sum(f_phi * h * w)
        C = np.abs(T) ** 2
        self.power_r = self.power_t + 10 * np.log10(C)
        power_r = 10 ** (self.power_r / 10)
        # self.power_r = self.power_t * np.abs(np.sum((self.h + self.g * self.Phi @ self.H) * self.w)) ** 2
        # reward = 10 * np.log2(10 ** (self.power_r / 10))
        # reward = 10 * np.log10(C)
        reward = 10 * np.log2(power_r)
        opt_reward = 0  # 占位符，用于理想奖励的计算
        return reward, opt_reward
    
    def AO0(self, h, H, g,):
        """Alternating Optimization算法"""
        habs = np.linalg.norm(h, axis=1, keepdims=True)
        w = h / habs

        rho = np.array([g.reshape(-1) * (H[i] @ w[i]) for i in range(self.I)])
        phi = np.angle(np.sum(rho, axis=0)) - np.angle(np.sum(h * w))
        Phi = np.exp(-1j * phi)
        
        r0, _ = self._compute_reward(h, H, g, w, Phi)

        for _ in range(1000):
            # wn = np.zeros_like(w)
            for i in range(self.I):
                tmp = h[i] + g.reshape(-1) * Phi @ H[i]
                w[i] = tmp / np.linalg.norm(tmp)
            # tmp = h + g * Phi @ H 
            # w = tmp / np.linalg.norm(tmp, axis=1, keepdims=True)
            # print(np.sum(wn-w))
            # w = wn
            rho = np.array([g.reshape(-1) * (H[i] @ w[i]) for i in range(self.I)])
            phi = np.angle(np.sum(rho, axis=0)) - np.angle(np.sum(h * w))
            Phi = np.exp(-1j * phi)
            
            r, _ = self._compute_reward(h, H, g, w, Phi) 
            if r - r0 < -1e6:
                raise ValueError("Reward is decreasing!")
            if np.abs(r - r0) < 1e-6:
                break
            r0 = r
            # print(r - r0, '\n') 
        return r, w, phi


    def AO_Low(self, h, H, g, sigma):
        """Alternating Optimization算法"""
        habs = np.linalg.norm(h, axis=1, keepdims=True)
        w = h / habs
        phi = np.exp(-1j * (- np.angle(np.sum(sigma * habs, axis=0, keepdims=True))))
        # r0, _ = self.compute_reward(h, H, g, sigma, w, phi)
        r0, _ = self._compute_reward(h, H, g, w, phi)
        fs0 = 0 
        for _ in range(100):
            f_phi = 1 + np.sum(phi * sigma, axis=1, keepdims=True)
            f_sigma = np.exp(-1j * np.angle(f_phi))
            w = w * f_sigma 
            phi = np.angle(np.sum(sigma * habs * f_sigma, axis=0, keepdims=True)) - np.angle(np.sum(habs * f_sigma))
            phi = np.exp(-1j * phi)
            # r, _ = self.compute_reward(h, H, g, sigma, w, phi) 
            r, _ = self._compute_reward(h, H, g, w, phi)
            # print(r, r0, r - r0,'\n', np.abs(f_sigma - fs0)) 
            if r - r0 < -1e6:
                raise ValueError("Reward is decreasing!")
            if np.abs(r - r0) < 1e-6 and (np.abs(f_sigma - fs0) < 1e-6).all():
                break
            r0 = r
            fs0 = f_sigma
        return r, w, phi


if __name__ == "__main__":
    env = RISSatComEnv(4, 16, 1, 3)
    sat_comm = RISSatCom.RISSatCom(0 , T=env.T, N=env.N, M=env.M, I=env.I)
    h, H, g, sigma = sat_comm.setup_channel()
    # env.AO(h, H, g, sigma)
    env.AO0(h, H, g)
    # print(h, H, g)
