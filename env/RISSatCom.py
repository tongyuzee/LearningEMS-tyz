import numpy as np


def aod(A, B, C):
    """计算信号出发角"""
    AB = np.array(B) - np.array(A)
    AC = np.array(C) - np.array(A)
    dot_product = np.sum(np.multiply(AB, AC), axis=1, keepdims=True)  # 按行点积
    cos_theta = dot_product / (np.linalg.norm(AB, axis=-1, keepdims=True) * np.linalg.norm(AC, axis=-1, keepdims=True))
    return np.arccos(cos_theta)


def aoa(V):
    """
    计算信号到达角
    返回相对于yOz平面上的方位角和俯仰角

    参数:
    V (numpy array): 信号到达点的坐标

    返回值:
    tuple: 包含方位角和俯仰角的元组，单位为弧度
    """
    V_x, V_y, V_z = np.hsplit(V, 3)
    magnitude = np.linalg.norm(V, axis=-1)
    azimuth = np.arctan2(V_z, V_y)
    elevation = np.arcsin(V_x / magnitude.reshape(-1, 1))
    return azimuth, elevation


class RISSatCom:
    def __init__(self,
                 episode_t,
                 T=1,
                 N=4,
                 I=2,
                 M=16,
                 G_S=40,
                 G_R=0,
                 G_T=0,
                 k=30):

        self.episode_t = episode_t

        self.T = T  # TR的天线数量
        self.N = N  # 卫星天线数量
        self.I = I  # 卫星数量
        self.M = M  # RIS的元素数量
        self.M1 = np.sqrt(self.M)
        self.M2 = np.sqrt(self.M)

        self.c = 3e8  # 光速，单位：米/秒
        self.f_c = 20e9  # 载波频率，单位：赫兹
        self.wavelength = self.c / self.f_c  # 计算波长

        self.G_S = G_S  # 卫星发射天线增益，单位：dBi
        self.G_R = G_R  # RIS增益，单位：dBi
        self.G_T = G_T  # 接收天线增益，单位：dBi
        self.k = 10 ** (k / 10)  # 莱斯因子

        self.RE = 6371E3  # 地球半径，单位：米
        self.h = 600e3  # 卫星高度，单位：米
        self.D = self.h + self.RE  # 卫星轨道半径，单位：米
        self.v = np.sqrt(3.986e14/self.D)  # 卫星的速度，单位：米/秒
        self.w = self.v / self.D  # 卫星的角速度，单位：弧度/秒
        self.theta0 = 55 * np.pi / 180  # 卫星初始位置，单位：弧度
        self.alpha = 5 * np.pi / 180  # 卫星的角度间隔，单位：弧度

        self.TT = np.round((np.pi - self.theta0 - self.theta0 - self.alpha) / self.w)  # 卫星飞行时间，单位：秒n

        self.l = 10  # RIS与TR之间的水平距离，单位：米
        self.hTR = 100  # TR高度，单位：米
        self.hRIS = 110  # RIS高度，单位：米
        self.delta = self.wavelength / 10  # RIS的元素间距，单位：米

        self.pTR = [0, 0, self.RE + self.hTR]
        self.pRIS = [self.l, 0, self.RE + self.hRIS]
        self.theta = np.zeros(self.I)
        self.pSAT = np.zeros((self.I, 3))

    def calculate_beta(self, G_X, G_Y, px, py):
        """计算电磁波传播的幅度增益。"""
        G_X = 10 ** (G_X / 10)
        G_Y = 10 ** (G_Y / 10)
        d_XY = np.linalg.norm(np.array(px) - np.array(py), axis=-1)
        return (self.wavelength * np.sqrt(G_X * G_Y)) / (4 * np.pi * d_XY)

    def current_position(self):
        """计算卫星的位置"""
        self.theta = np.arange(self.I)*self.alpha + self.theta0 + self.w * self.episode_t
        self.pSAT = [[0, self.D * np.cos(x), self.D * np.sin(x)] for x in self.theta]

        return self.pSAT

    def setup_channel(self):
        """设置信道并计算相关参数。"""
        self.pSAT = self.current_position()
        beta_ts = self.calculate_beta(self.G_S, self.G_T, self.pTR, self.pSAT)
        beta_rs = self.calculate_beta(self.G_S, self.G_R, self.pRIS, self.pSAT)
        beta_tr = self.calculate_beta(self.G_T, self.G_R, self.pTR, self.pRIS)

        # # 暂时不考虑幅度增益
        # beta_ts = beta_ts / beta_ts
        # beta_rs = beta_rs / beta_rs
        # beta_tr = beta_tr / beta_tr

        hNLOS = np.random.normal(0, np.sqrt(0.5), (self.I, self.N)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.I, self.N))
        HNLOS = np.random.normal(0, np.sqrt(0.5), (self.I, self.M, self.N)) + 1j * np.random.normal(0, np.sqrt(0.5), (self.I, self.M, self.N))

        aoa_rs_a, aoa_rs_e = aoa(np.array(self.pRIS) - np.array(self.pSAT))
        aod_ts = aod(self.pSAT, self.pTR, [0, 0, 0])
        aod_rs = aod(self.pSAT, self.pRIS, [0, 0, 0])

        aoa_z = self.f_sv(self.M2, self.delta, np.sin(aoa_rs_e) * np.sin(aoa_rs_a))
        aoa_y = self.f_sv(self.M1, self.delta, np.sin(aoa_rs_e) * np.cos(aoa_rs_a))
        HLOS_1 = np.array([np.kron(aoa_z[i], aoa_y[i]) for i in range(self.I)])  # size:(I, M=M2*M1)

        HLOS_2 = self.f_sv(self.N, self.wavelength / 2, np.sin(aod_rs))  # size:(I, N)

        HLOS = np.array([HLOS_1[i].reshape(-1, 1) @ HLOS_2[i].reshape(1, -1) for i in range(self.I)])  # size:(I, M, N)

        hLOS = self.f_sv(self.N, self.wavelength / 2, np.sin(aod_ts))  # size:(I, N)
        hh = beta_ts.reshape(-1, 1) * (np.sqrt(self.k / (1 + self.k)) * hLOS + np.sqrt(1 / (1 + self.k)) * hNLOS)

        Htmp = np.sqrt(self.k / (1 + self.k)) * HLOS + np.sqrt(1 / (1 + self.k)) * HNLOS
        HH = np.array([beta_rs[i]*Htmp[i] for i in range(self.I)])

        d_tr = np.linalg.norm(np.array(self.pTR) - np.array(self.pRIS))
        gg = beta_tr * np.exp(-1j * 2 * np.pi * d_tr / self.wavelength) * np.array(np.ones(self.M))
        
        sigma = np.array([beta_rs[i] / beta_ts[i] * gg * HLOS_1[i] for i in range(self.I)])
        return hh, HH, gg.reshape(1, -1), sigma

    def f_sv(self, k, l, gamma):
        """生成导向向量"""
        k_vals = np.arange(int(k))  # 创建从 0 到 k-1 的数组
        const = -1j * 2 * np.pi * l / self.wavelength  # 提取常量部分
        return np.exp(const * k_vals * gamma)  # 使用广播进行矢量化计算
        # return np.array([[np.exp(-1j * 2 * np.pi * l * i * g / self.wavelength) for i in range(int(k))] for g in np.hstack(gamma)])
    
    # def AO(self, h, sigma):
    #     """Alternating Optimization算法"""
    #     habs = np.linalg.norm(h, axis=1, keepdims=True)
    #     w = h / habs
    #     phi = np.exp(-1j * np.angle(np.sum(sigma * habs, axis=0)))
    #     r0, _ = self._compute_reward(w, phi)
    #     for _ in range(1000):
    #         f_phi = 1 + np.sum(phi * sigma, axis=1, keepdims=True)
    #         f_sigma = np.exp(-1j * np.angle(f_phi))
    #         w = w * f_sigma 
    #         phi = np.angle(np.sum(sigma * habs * f_sigma, axis=0)) - np.angle(np.sum(habs * f_sigma))
    #         phi = np.exp(1j * phi)
    #         r, _ = self._compute_reward(w, phi) 
    #         if r - r0 < 1e-6:
    #             break
    #         pass


# if __name__ == "__main__":
#     sat_comm = RISSatCom(10)
#     h, H, g, sigma = sat_comm.setup_channel()
#     sat_comm.AO(h, sigma)
#     print(h, H, g)
