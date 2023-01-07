##### Code for Offline Reinforcement Learning with Uncertainty Critic Regularization Based on Density Estimation.

# 1.Installation

To run experiments, you will need to install the following packages preferably in a conda virtual environment

- gym==0.23.1
- mujoco_py
- torch==1.13.0
- tqdm==4.64.0
- tensorboardX==2.5.1
- scipy==1.8.0
- numpy==1.22.3
- d4rl

Suggested build environment:

~~~shell
conda create -n density-ucr python=3.8
conda activate density-ucr
pip install -r requirements.txt
#install d4rl
pip install git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl
~~~

# 2.Usage

To run the code with the default parameters, simply execute the following command:

## 2.1 Pre-training VAE model
Pre-train the density model for each task, such as
```shell
python train_vae.py --task=walker2d --ds=medium --gpu-no=0 
python train_vae.py --task=halfcheetah --ds=medium --gpu-no=0 
python train_vae.py --task=hopper --ds=medium --gpu-no=0 
```
```shell
python train_vae.py --task=maze2d --ds=umaze-dense --gpu-no=0 
python train_vae.py --task=maze2d --ds=medium-dense --gpu-no=0 
python train_vae.py --task=maze2d --ds=large-dense --gpu-no=0 
```

After the density model training is completed, store the trained model in **model/mujoco** or **model/maze2d** according to different tasks. 

Then write the path of the corresponding model into the configuration file **(config)**.
## 2.2 Offline RL
Run the following command to train offline RL on D4RL.
```shell
python run.py --algo=density-ucr-offline --task=walker2d --ds=medium --gpu-no=0 --seed=10 --save=True --train=True
python run.py --algo=density-ucr-offline --task=halfcheetah --ds=medium --gpu-no=0  --seed=10 --save=True --train=True
python run.py --algo=density-ucr-offline --task=hopper --ds=medium --gpu-no=0  --seed=10 --save=True --train=True
```

```shell
python run_maze2d.py --algo=density-ucr-offline --task=maze2d --ds=umaze-dense --gpu-no=0 --seed=10 --save=True --train=True
python run_maze2d.py --algo=density-ucr-offline --task=maze2d --ds=medium-dense --gpu-no=0  --seed=10 --save=True --train=True
python run_maze2d.py --algo=density-ucr-offline --task=maze2d --ds=large-dense --gpu-no=0  --seed=10 --save=True --train=True
```
## 2.3 Online Fine-tuning
* Put the model trained in 2.2 into the location corresponding to **model/policy**, and write the path of the corresponding model into the configuration file (config).


* Run the following command to online fine-tune on Mujoco Gym with offline models.

```shell
python run.py --algo=density-ucr-online --task=walker2d --ds=medium --gpu-no=0 --seed=10 --save=True --train=True
python run.py --algo=density-ucr-online --task=halfcheetah --ds=medium --gpu-no=0  --seed=10 --save=True --train=True
python run.py --algo=density-ucr-online --task=hopper --ds=medium --gpu-no=0  --seed=10 --save=True --train=True
```

## 2.4 Logging
This codebase uses tensorboard. You can view saved runs with:

```shell
tensorboard --logdir=results/logs/<task>
```
## Contact

If you have any question, please contact lichao.winner@gmail.com.

## The other Baseline code implementations are as follows：
| Algorithm | code                                              |
|-----------| ------------------------------------------------- |
| BC        |  https://github.com/tinkoff-ai/CORL             |
| AWAC      | https://github.com/ikostrikov/jaxrl               |
| IQL       | https://github.com/ikostrikov/implicit_q_learning |
| CQL       | https://github.com/tinkoff-ai/CORL                |
| TD3+BC    | https://github.com/sfujim/TD3_BC         |
| EDAC      | https://github.com/tinkoff-ai/CORL        |
| PBRL      | https://github.com/baichenjia/pbrl          |

