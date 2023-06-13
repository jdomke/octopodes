#ifndef GRIDWORLD_H 
#define GRIDWORLD_H
#include "EnvironmentsBase.hpp"
#include <random>
#include <cmath>

#include <algorithm>     
namespace octorl {

    class Gridworld : virtual public EnvironmentsBase {
        private:
            static const int action_space_size = 4;
            static const int observation_space_size = 15;
        public:
            Gridworld();


    };
}

#endif