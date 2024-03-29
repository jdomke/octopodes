#ifndef ENVIRONMENTS_BASE_H
#define ENVIRONMENTS_BASE_H
#include "Spaces.hpp"


namespace octorl {

    class EnvironmentsBase {
        public:
            virtual StepReturn step(int action) = 0;
            virtual StepReturn reset() = 0;
            virtual int getActionSize() = 0;
            // gonna have to update for higher then 1d input dim
            virtual int getObservationSize() = 0;
            virtual int currentStep() = 0;
            virtual int memorySize() = 0;
            virtual torch::Tensor getState() = 0;
            virtual torch::Tensor shapeObservation(torch::Tensor buffer) = 0;
            virtual torch::Tensor obsBuffer(int b) = 0;

        private:
            static const int action_space_size;
            static const int observation_space_size;
        protected:

    };

}

#endif