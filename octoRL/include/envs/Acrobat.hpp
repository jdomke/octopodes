#ifndef ACROBAT_H
#define ACROBAT_H


#include "EnvironmentsBase.hpp"
#include <random>
#include <cmath>
#include <algorithm>      
namespace octorl {
    const double pi = 3.14159265358979323846;
    class Acrobat : virtual public EnvironmentsBase {
        private:
            float theta1_cos;
            float theta1_sin;
            float theta2_cos;
            float theta2_sin;
            float theta1_ang;
            float theta2_ang;
            int steps;
            static const int action_space_size = 3;
            static const int observation_space_size = 5;
            float dt = 0.2;

            const float LINK_LENGTH_1 = 1.0;  // [m]
            const float LINK_LENGTH_2 = 1.0;  //# [m]
            const float LINK_MASS_1 = 1.0;  //#: [kg] mass of link 1
            const float LINK_MASS_2 = 1.0;  //#: [kg] mass of link 2
            const float LINK_COM_POS_1 = 0.5; // #: [m] position of the center of mass of link 1
            const float LINK_COM_POS_2 = 0.5;//  #: [m] position of the center of mass of link 2
            const float LINK_MOI = 1.0; // #: moments of inertia for both links

            const float MAX_VEL_1 = 4 * pi;
            const float MAX_VEL_2 = 9 * pi;

            void transition(int action);
        public:
            Acrobat(int seed); 
            Acrobat(); 
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
            torch::Tensor getState();




    }


}
#endif