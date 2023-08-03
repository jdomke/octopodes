#ifndef CNN_TEST_H
#define CNN_TEST_H

#include "EnvironmentsBase.hpp"
#include <random>
#include <cmath>
#include <algorithm>      

namespace octorl {

    class CNNTest : virtual public EnvironmentsBase {
        public:
            CNNTest();
             CNNTest(int s);
            std::default_random_engine gen;
            std::uniform_real_distribution<float> distribution{std::uniform_real_distribution<float>(-0.6,-0.4)};
            StepReturn step(int action);
            StepReturn reset();
            int getActionSize();
            int getObservationSize();
            int currentStep();
            int memorySize() {return 2*(3*32*32) + 3;}
            torch::Tensor getState();
            torch::Tensor shapeObservation(torch::Tensor buffer);
            torch::Tensor obsBuffer(int b);
        private:
            static const int action_space_size = 3;
            static const int observation_space_size = 32*32*3;
            int steps;
            int seed;
    };
}

#endif