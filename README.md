# Mutual Information-based State-Control (MISC)

Here is the code for our paper "Mutual Information-based State-Control for Intrinsically Motivated Reinforcement Learning".

The code was developed by Rui Zhao during a research internship at Horizon Robotics Inc. Cupertino, CA, USA.

The paper is under review at a conference.

The preprint version is available on arXiv: https://arxiv.org/abs/2002.01963.

Our code is based on OpenAI Baselines (link: https://github.com/openai/baselines).   

## Learned Control Behaviors without Task Rewards

<img width="300" height="" src="/demos/reach.gif"> <img width="300" height="" src="/demos/push.gif">
<img width="300" height="" src="/demos/slide.gif"> <img width="300" height="" src="/demos/pick.gif">

## Prerequisites  

The code requires python3 (>=3.5) with the development headers. You'll also need system packages CMake, OpenMPI and zlib. Those can be installed as follows  

### Usage  
    
```bash
sudo apt-get update && sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev
```

To run the code, you need to install OpenAI Gym (link: https://github.com/openai/gym).  
We use the robotics environment in OpenAI Gym, which needs the MuJoCu physics engine (link: http://www.mujoco.org/).   

The experiments were carried out on a 16-CPUs server.  
We use 16 CPUs for training.  
If you are running the experiments on a laptop, please configure a smaller number of CPUs.  
Note that, with less CPUs, the performance will be affected.  

After the installation of dependencies, you can start to reproduce the experimental results by running the following commands.

The following command is for training an agent using MISC without any rewards or supervision.
```
python baselines/her/experiment/train.py --env_name FetchPickAndPlace-v1 --n_epochs 50 --num_cpu 16 --logging True --seed 0 --note SAC+MISC
```
To test the learned policies, you can run the command:  
```
python baselines/her/experiment/play.py /path/to/an/experiment/policy_latest.pkl --note SAC+MISC
```
The rendered video is saved alongside the policy file.

You can also extract the weight from the saved TensorFlow graph .pkl file.
To do this, you need to put the path of the .pkl file into the config file, for example, into "params/SAC+MISC.json".
Afterwards, you can use the following command to convert the .pkl file:
```
python baselines/her/experiment/save_weight.py --env_name FetchPickAndPlace-v1 --note SAC+MISC-r --seed 0
```

After converting, you can use the pre-trained MI discriminator to accelerate learning:
```
python baselines/her/experiment/train.py --env_name FetchPickAndPlace-v1 --n_epochs 50 --num_cpu 16 --logging True --seed 0 --note SAC+MISC-r
```

## Citation

Citation of the paper:

```
@article{zhao2020mutual,
  title={Mutual Information-based State-Control for Intrinsically Motivated Reinforcement Learning},
  author={Zhao, Rui and Tresp, Volker and Xu, Wei},
  journal={arXiv preprint arXiv:2002.01963},
  year={2020}
}
```

## Licence

MIT
