#ifndef MOUNTAIN_CAR_H
#define MOUNTAIN_CAR_H

#include "EnvironmentsBase.hpp"
#include <random>
#include <cmath>
#include <algorithm>      
namespace octorl {

    class MountainCar : virtual public EnvironmentsBase {
        private:
            //DiscreteActionSpace action_space;
            const float FORCE = 0.001;
            const float GRAVITY = 0.0025;
            float position;
            float velocity;
            int steps;
            static const int action_space_size = 3;
            static const int observation_space_size = 2;
            void transition(int action);
        public:
            MountainCar(int seed); 
            MountainCar(); 
            std::default_random_engine gen;
            std::uniform_real_distribution<float> distribution{std::uniform_real_distribution<float>(-0.6,-0.4)};
            StepReturn step(int action);
            StepReturn reset();
            float getPosition();
            int getActionSize();
            int getObservationSize();
            int currentStep();
            void setState(float p, float v);
            int memorySize();
    };
}

#endif