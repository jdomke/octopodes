#octorl

## Build
### Requirements
Pytorch installed on system (either through pip or building from source)
A compiler supporting MPI
PAPI
libconfig

### Build commands
```bash

mkdir build
cd build
# pytorch installed through pip
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
# or
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..

cmake --build . --config Release
```

## Configuration Files
OctoRL uses the libconfig library to handle configuration files.
There are two fields required for a configuration file
1. environment
1. some form of agent configuration (A3C, A2C, or DqnAsync curently)

### Environment configuration
To set the environment include 
```
environment = "somme supported environment"
```
currently Cartpole and MountainCar are supported

### Agent configuration
Each agent requires the definition of some policy
A3C and A2C require both an actor and critic policy, while DqnAsync just requires the definition of a policy.

A policy can be defined by a collection of layers. Each layer requires a type, a label, and an activation function.
You can also define the layers input size, output size, kernel_size, padding, stride, and dilation (when applicable)

#### Supported Types
1. linear
1. conv2d
1. flatten
1. max_pool_2d

#### Supported Activation Functions
1. relu
1. softmax
1. none (linear)
1. sigmoid
1. tanh

#### A3C and A2C setable training parameters
1. Buffer_size - integer - default = 100000
1. Episode_count - integer - default = 500
1. Seed - integer
1. Batch_size - integer - default = 32
1. Learning_rate - float - default = 0.001
1. Gamma - float - default = 0.99

#### DqnAsync setable training parameters
1. Buffer_size - integer - default = 100000
1. Episode_count - integer - default = 500
1. Seed - integer
1. Batch_size - integer - default = 32
1. Batch_freq - integer - default = 32
1. Learning_rate - float - default = 0.001
1. Gamma - float - default = 0.99
1. Epsilon - float - default = 1
1. Epsilon_min - float - default = 0.01
1. Epsilon_decay - float - default = 0.95

See files in configs folder for example configuration files for each Agent type

To run code 
```bash
# where P is the number of nodes and Conf is the path to your configuration file
mpirun -n P main Conf
```