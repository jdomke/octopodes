#ifndef CARTPOLE_H
#define CARTPOLE_H


#include "EnvironmentsBase.hpp"
#include <random>
#include <cmath>
#include <limits>
// https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py#L149
namespace octorl {
    const double pi = 3.14159265358979323846;
    class Cartpole : virtual public EnvironmentsBase {
        private: 
            const float GRAVITY = 9.8;
            const float MASSCART = 1.0;
            const float MASSPOLE = 0.1;
            const float TOTAL_MASS = MASSCART + MASSPOLE;
            const float LENGTH = 0.5;
            const float POLEMASS_LENGTH = MASSPOLE * LENGTH;
            const float FORCE_MSG = 10;
            const float TAU = 0.02;

            const double THETA_THRESHOLD_RADIANS = 12 * 2 * pi/360;
            const float X_THRESHOLD = 2.4;

            static const int action_space_size = 2;
            static const int observation_space_size = 4;

            float x;
            float x_dot;
            float theta;
            float theta_dot;
            float x_limit = X_THRESHOLD * 2;
            float theta_limit = THETA_THRESHOLD_RADIANS * 2;
            int steps;

            std::default_random_engine gen;
            std::uniform_real_distribution<float> position_gen{std::uniform_real_distribution<float>(-x_limit, x_limit)};
            std::uniform_real_distribution<float> velocity_gen{std::uniform_real_distribution<float>(std::numeric_limits<float>::min(), std::numeric_limits<float>::max())};
            std::uniform_real_distribution<float> theta_gen{std::uniform_real_distribution<float>(-theta_limit, theta_limit)};
            std::uniform_real_distribution<float> starting_gen{std::uniform_real_distribution<float>(-0.05,0.05)};



        public:
            Cartpole();
            Cartpole(int seed);
            void setState(float xset, float x_dotset, float thetaset, float theta_dotset);
            StepReturn step(int action);
            StepReturn reset();
            float getPosition();
            int getActionSize();
            int getObservationSize();
            int currentStep();
            int memorySize();
            torch::Tensor getState();

            

    };
}

#endif