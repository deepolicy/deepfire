# deepfire

Implementation of the method described in our paper:

DeepFire: A Network for Multi Agents with Hybrid Action Space

# Abstract

This paper is primarily involved in a novel neural network, which intends to address the control of multi-agent, hybrid-action space large-scale strategy games. In view of the change in the number of agents, in the input layer, one-dimensional convolution and global max pool are employed to convert any number of inputs into a vector, thereby accepting an uncertain number of agent observation inputs; in the output layer, the information of a single agent is combined with the overall situation to output its action, so as to realize the control of all agents in a forward propagation; during training, Proximal Policy Optimization (PPO) and the distribution mask are utilized to construct a united objective function to achieve joint training of multi-agent with hybrid action distribution. Through the training experiment of 21 parallel games, the neural network can adapt to the game control of a varying number of agents, and the total reward per episode witnesses a rise as the number of learning episodes increases. Most importantly, all the codes at https://github.com/deepolicy/deepfire are available, allowing researchers to utilize higher-performance computing platforms to parallel more environments to gain better results.

For more support, please contact with zhouxin12@nudt.edu.cn

![deepolicy](https://github.com/deepolicy/deepfire/blob/master/bf-nn3-0909-1945-1e-5-bs128-1217.reward.png)

![deepolicy](https://github.com/deepolicy/deepfire/blob/master/assets/battle-field.png)
