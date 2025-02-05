# 调试日志

## AO

### 2025/01/13

卫星单天线场景效果符合预期

![单天线场景效果](./../../../../02workspace/LearningEMS-tyz/Learning_Curves/AO/1_10000_3_120_0.99_0.999_5e-05_5e-04_seed00024_20250113_200855.png)

多天线场景依然有问题

![多天线场景效果](./../../../../02workspace/LearningEMS-tyz/Learning_Curves/AO/4_10000_2_120_0.99_0.999_5e-05_5e-04_seed00024_20250113_195112.png)

怀疑：AO算法对w的计算未达到最优解

### 2025/01/20

多天线场景优化完成，问题找到为计算信道增益公式中缺少了**w的共轭处理（np.conj(w)）**！

最终，优化过后的结果如图所示

卫星天线数 N=4，RIS元素数 M=10000，共轨卫星数 I=3：

![多天线场景效果](./../../../../02workspace/LearningEMS-tyz/Learning_Curves/AO/4_10000_3_120_0.99_0.999_5e-05_5e-04_seed00024_20250120_210603.png)

卫星天线数 N=16，RIS元素数 M=1600，共轨卫星数 I=3：

![多天线场景效果](./../../../../02workspace/LearningEMS-tyz/Learning_Curves/AO/16_1600_3_120_0.99_0.999_5e-05_5e-04_seed00024_20250120_211640.png)

### 2025/01/25

修改AO算法，将共轭处理conj转移到AO算法内部

![多天线场景效果](./../../../../02workspace/LearningEMS-tyz/Learning_Curves/AO/16_1600_3_120_0.99_0.999_5e-05_5e-04_seed00024_20250125_144747.png)

## SAC

### 2025/01/16

修改gamme=0.9（原来为0.99），奖励函数效果好一些，但是依然会下降。

![gamme=0.9](./../../../../02workspace/LearningEMS-tyz/Learning_Curves/SAC/2_4_2_120_0.9_1e-04_1e-03_1e-03_seed00516_20250116_111155.png)

修改学习率为2e-4, 2e-3, 2e-3

![学习率调整到2e-4,2e-3](./../../../../02workspace/LearningEMS-tyz/Learning_Curves/SAC/2_4_2_120_0.9_2e-04_2e-03_2e-03_seed00516_20250116_164909_eps_3699.png)

修改学习率为5e-4, 5e-3, 5e-3

![学习率调整到5e-4,5e-3](./../../../../02workspace/LearningEMS-tyz/Learning_Curves/SAC/2_4_2_120_0.9_5e-04_5e-03_5e-03_seed00516_20250116_183957_eps_3999.png)

SAC与AO算法的比较如图

![SAC VS AO](./../../../../02workspace/LearningEMS-tyz/Learning_Curves/SAC/2_4_2_120_0.9_5e-04_5e-03_5e-03_seed00516_20250116_183957_compare.png)

**目前AO算法的总和奖励为539.85，但是SAC的总和奖励仅仅为-1386.47，相差很远。**

### 2025/01/22

修改了信道增益计算公式bug后，重新训练了SAC算法，并且与修改后的AO算法进行比较结果如图：

![修改信道增益bug](./../../../../02workspace/LearningEMS-tyz/Learning_Curves/SAC/2_4_2_120_0.9_5e-05_5e-04_5e-04_seed00516_20250121_001638_eps_6599.png)

![SAC VS AO](./../../../../02workspace/LearningEMS-tyz/Learning_Curves/SAC/2_4_2_120_0.9_5e-05_5e-04_5e-04_seed00516_20250121_001638_compare.png)

**目前AO算法的总和奖励为1316.65，但是SAC的总和奖励仅仅为-1400，相差很远。**

### 2025/02/05

将SAC算法动作约束从[0,1]改为[-1,1]

![奖励曲线](./../../../../02workspace/LearningEMS-tyz/Learning_Curves/SAC/2_4_2_120_0.95_1e-04_5e-04_5e-04_seed00516_20250205_191032_eps_0299.png)

![SAC VS AO](./../../../../02workspace/LearningEMS-tyz/Learning_Curves/SAC/2_4_2_120_0.95_1e-04_5e-04_5e-04_seed00516_20250205_191032_compare.png)

**目前AO算法的总和奖励为1316.65，但是SAC的总和奖励仅仅为1200左右，还需要继续收敛。**
