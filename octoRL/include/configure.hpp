#ifndef CONFIGURE_H
#define CONFIGURE_H

#include <map>
#include "Policy.hpp"
#include <libconfig.h++>
#include <string>

#include "mpi.h"
#include "envs/MountainCar.hpp"
#include "envs/Cartpole.hpp"
#include "envs/CNNTest.hpp"
#include "agents/DqnAsync.hpp"
#include "agents/A3C.hpp"
#include "agents/A2C.hpp"

void configureAndRun(const char* file);
void a2cConfigureAndRun(const libconfig::Setting& root);
void a3cConfigureAndRun(const libconfig::Setting& root);
void dqnAsyncConfigureAndRun(const libconfig::Setting& root);
octorl::LayerInfo layerParse(const libconfig::Setting& layer);
octorl::Policy modelParse(const libconfig::Setting& model, int s);

#endif