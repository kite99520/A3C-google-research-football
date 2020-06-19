# A3C-google-research-football
A simple A3C algorithm for google research football

## 算法
采用Asynchronous Advantage Actor-critic(简称A3C)算法，基本思想：结合value-based和policy-based，
使用神经网络同时输出𝜋(𝑠,𝑎)和𝑉(𝑠)。维护一个全局Critic网络的同时，并行的构建多个Agent，每个Agent有独立的一套环境和局部网络，
采样得到的gradient在全局网络上进行共享和更新，并定期把全局网络的参数更新到每个Agent的局部网络上，这样既节约了DQN算法中ReplayBuffer所需的存储，
也减弱了采样的相关性。

![算法](https://github.com/kite99520/A3C-google-research-football/blob/master/graph/p1.png)

## 具体实现
**Parallel：**  采用pytorch中的multiprocess机制，多进程实现Agent采样。

**Network：**
用4个(3,3)卷积核完成特征提取，(16,72,96)->(32,5,6)，后接一个lstmcell，隐层输出结果分别通过两个全连接层输出策略π和值函数value。使用lstmcell出于输入state带有时间序列性质的考量，训练时发现效果似乎优于普通的全连接层。
input(16,72,96)->[conv((3,3),padding=1)->relu]*4->lstmcell((32,5,6),512)->hidden(512)

hidden->linear(512,action.space.n)->π

hidden->linear(512,1)->value

**Loss：**
关于policy，采用GAE(generalized advantage estimator)，对优势函数A进行一定程度的加权作为GAE。

![](http://chart.googleapis.com/chart?cht=tx&chl=$$L_{\pi}=-\sum_{t=1}^{\infty}A_t\nabla\log\pi_{\theta}(s|a)$$)

关于value,

![](http://chart.googleapis.com/chart?cht=tx&chl=$$L_v=\sum_{i=1}^{n}e_{i}^2$$)

关于regularization，适当增加其他action的探索几率，避免样本过于集中。

![](http://chart.googleapis.com/chart?cht=tx&chl=$$entropy_i=-\pi_{\theta}\log\pi_{\theta}$$)

![](http://chart.googleapis.com/chart?cht=tx&chl=$$L_{reg}=-\sum_{i=1}^{n}entropy_i$$)


## 实验环境与运行

**gfootball环境配置**
```
!apt-get update
!apt-get install libsdl2-gfx-dev libsdl2-ttf-dev
!git clone -b v2.0.7 https://github.com/google-research/football.git
!mkdir -p football/third_party/gfootball_engine/lib
!wget https://storage.googleapis.com/gfootball/prebuilt_gameplayfootball_v2.0.7.so -O football/third_party/gfootball_engine/lib/prebuilt_gameplayfootball.so
!cd football && GFOOTBALL_USE_PREBUILT_SO=1 pip3 install .
```
**目录**
src文件夹中包含model.py、optimizer.py、process.py，分别描述神经网络配置、优化器设置和单个进程的训练算法。trained_models文件夹包含训练好的模型参数，其中params.pkl是γ=0.99是训练得到的结果，params2.pkl是γ=0.992的结果。

**训练**
```
!python train.py
```

```
--env_name  环境名称，默认'academy_3_vs_1_with_keeper'
--lr  学习率，默认1e-4
--eps optimizer中的参数，防止出现0/0的情况，默认1e-5
--lr_decay  学习率是否衰减，默认False
--gamma 折扣因子γ，默认0.99
--tau GAE中的参数λ，默认1.0
--beta  Entropy Loss项的权重β，默认0.01
--num_local_steps 每个agent采样时，每个episode的最大step数，默认128
--num_global_steps  一次训练，每个agent能进行的总的最大step数，默认2e6
--num_processes 进程数，即用于并行采样的agent数量，默认6
--save_interval 两次保存之间间隔的episode数，默认50
--print_interval  两次输出训练时间，平均reward之间间隔的episode数，默认50
--save_path 保存模型参数的文件夹，默认"trained_models"
--save_path_file  保存模型参数的具体路径，如"/content/drive/My Drive/A3C-pytorch-master/trained_models/params2.pkl"
--load_from_previous_stage  是否从之前已保存的模型加载，默认False
--use_gpu 是否使用gpu进行加速，默认False
```

**测试**
```
!python test.py
```
```
--env_name  环境名称，默认'academy_3_vs_1_with_keeper'
--load_path 加载模型参数的具体路径，如"/content/drive/My Drive/A3C-pytorch-master/trained_models/params2.pkl"
--play_episodes 测试的episode数，默认2000
```











