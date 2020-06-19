# A3C-google-research-football
A simple A3C algorithm for google research football

![](http://chart.googleapis.com/chart?cht=tx&chl=$a_2$)

采用Asynchronous Advantage Actor-critic(简称A3C)算法，基本思想：结合value-based和policy-based，
使用神经网络同时输出𝜋(𝑠,𝑎)和𝑉(𝑠)。维护一个全局Critic网络的同时，并行的构建多个Agent，每个Agent有独立的一套环境和局部网络，
采样得到的gradient在全局网络上进行共享和更新，并定期把全局网络的参数更新到每个Agent的局部网络上，这样既节约了DQN算法中ReplayBuffer所需的存储，
也减弱了采样的相关性。

