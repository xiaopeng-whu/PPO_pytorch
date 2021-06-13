# PPO_pytorch
---
This is a simple implement of [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)**（PPO）** using PyTorch.


### To do list
---
* Migrate to linux platform to complete testing in [roboschool](https://github.com/openai/roboschool) environment.
* Use GAE(Generalized Advantage Estimation) instead of monte-carlo estimate.

### How to run
---
* Clone repository :
```
$ git clone https://github.com/xiaopeng-whu/PPO_pytorch.git 
$ cd PPO_pytorch
```
- To train a new network : run `train.py`
- To test a preTrained network : run `test.py`
- To plot graphs using log files : run `plot_graph.py`
- To save images for gif and make gif using a preTrained network : run `make_gif.py`
- All parameters and hyperparamters to control training / testing / graphs / gifs are in their respective `.py` file
- All the **hyperparameters used for training (preTrained) policies are listed** in the [`README.md` in PPO_preTrained directory](https://github.com/nikhilbarhate99/PPO-PyTorch/tree/master/PPO_preTrained)

### Results
---
* BipedalWalker-v2

![](/PPO_gifs/BipedalWalker-v2/PPO_BipedalWalker-v2_gif_0.gif)

![](/PPO_figs/BipedalWalker-v2/PPO_BipedalWalker-v2_fig_0.png)

### Dependencies
---
Trained and Tested on:
- Python 3
- PyTorch
- NumPy
- gym==0.15.4(gym==0.10.5 may encounter bugs when making gif.)

Training Environments 
- Box-2d [(No module named ‘Box2D’ 解决方案)](https://ithelp.ithome.com.tw/articles/10229349)
- Roboschool [(It doesn't supporting windoes. So change it to linux or osx machine environment)](linux or osx machine)
- pybullet

Graphs and gifs
- pandas
- matplotlib
- Pillow

### References
---
[PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch)

[强化学习经典算法笔记(十二)：近端策略优化算法（PPO）实现，基于A2C（下）](https://blog.csdn.net/hhy_csdn/article/details/107043832)

[PPO-for-Beginners](https://github.com/ericyangyu/PPO-for-Beginners)([配套教程](https://medium.com/@eyyu/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8.))
