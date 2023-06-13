#ifndef CARTPOLE_H
#define CARTPOLE_H

#include "EnvironmentsBase.hpp"
#include <random>
#include <cmath>

namespace octorl {

    class Cartpole : virtual public EnvironmentsBase {
        private: 
            const float GRAVITY = 9.8;
            const float MASSCART = 1.0;
            const float MASSPOLE = 0.1;
            const float TOTALMASS = MASSCART + MASSPOLE;



    };
}

#endif