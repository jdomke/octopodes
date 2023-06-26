#ifndef DRIVER_H
#define DRIVER_H
#include <iostream>
#include <time.h>
#include "envs/MountainCar.hpp"
#include "envs/Blackjack.hpp"
#include "envs/Cartpole.hpp"
#include <omp.h>
#include <time.h>
#include <unistd.h>
//#include "Mlp.hpp"
#include "agents/Dqn.hpp"

void mountainCarDQNTest();
void blackjackDQNTest();
void cartpoleDQNTest();
void openMpFirstTry();

#endif