import gymnasium as gym
import random
env = gym.make('CartPole-v0')
f = open("cartpoleTest.csv", 'w')

f.write("x,xdot,theta,thetadot,action,x1,xdot1,theta1,thetadot1\n")

for i in range(1000):
    env.reset()
    f.write(f"{env.state[0]},{env.state[1]},{env.state[2]},{env.state[3]},")
    action = random.randint(0,1)
    observation, reward, terminated, truncated, info = env.step(action)
    f.write(f"{action},{env.state[0]},{env.state[1]},{env.state[2]},{env.state[3]}\n")
    while(not terminated):
        f.write(f"{env.state[0]},{env.state[1]},{env.state[2]},{env.state[3]},")
        action = random.randint(0,1)
        observation, reward, terminated, truncated, info = env.step(action)
        f.write(f"{action},{env.state[0]},{env.state[1]},{env.state[2]},{env.state[3]}\n")

f.close()

env = gym.make('MountainCar-v0')
f = open("MountainCarTest.csv", 'w')
f.write("position,velocity,action,position1,velocity1\n")

for i in range(1000):
    env.reset()
    f.write(f"{env.state[0]},{env.state[1]},")
    action = random.randint(0,2)
    count = 0 
    observation, reward, terminated, truncated, info = env.step(action)
    f.write(f"{action},{env.state[0]},{env.state[1]}\n")
    while(count < 10):
        f.write(f"{env.state[0]},{env.state[1]},")
        action = random.randint(0,2)
        observation, reward, terminated, truncated, info = env.step(action)
        f.write(f"{action},{env.state[0]},{env.state[1]}\n")
        count += 1
f.close()
