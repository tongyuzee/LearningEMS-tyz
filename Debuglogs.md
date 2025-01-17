# 调试日志

## AO

2025/01/13

卫星单天线场景效果符合预期

![单天线场景效果](./../../../../02workspace/LearningEMS-tyz/Learning_Curves/AO/1_10000_3_120_0.99_0.999_5e-05_5e-04_seed00024_20250113_200855.png)

多天线场景依然有问题

![多天线场景效果](./../../../../02workspace/LearningEMS-tyz/Learning_Curves/AO/4_10000_2_120_0.99_0.999_5e-05_5e-04_seed00024_20250113_195112.png)

怀疑：AO算法对w的计算未达到最优解

## SAC

2025/01/16

修改gamme=0.9（原来为0.99），奖励函数效果好一些，但是依然会下降。

![gamme=0.9](./../../../../02workspace/LearningEMS-tyz/Learning_Curves/SAC/2_4_2_120_0.9_1e-04_1e-03_1e-03_seed00516_20250116_111155.png)

修改学习率为2e-4, 2e-3, 2e-3

![学习率调整到2e-4,2e-3](./../../../../02workspace/LearningEMS-tyz/Learning_Curves/SAC/2_4_2_120_0.9_2e-04_2e-03_2e-03_seed00516_20250116_164909_eps_3699.png)

修改学习率为5e-4, 5e-3, 5e-3

![学习率调整到5e-4,5e-3](./../../../../02workspace/LearningEMS-tyz/Learning_Curves/SAC/2_4_2_120_0.9_5e-04_5e-03_5e-03_seed00516_20250116_183957_eps_3999.png)

SAAC与AO算法的比较如图

![SAC VS AO](./../../../../02workspace/LearningEMS-tyz/Learning_Curves/SAC/2_4_2_120_0.9_5e-04_5e-03_5e-03_seed00516_20250116_183957_compare.png)

**目前AO算法的总和奖励为539.85，但是SAC的总和奖励仅仅为-1386.47，相差很远。**
