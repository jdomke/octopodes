// maybe safe to just use floats as input
// may want to add high/low span
#ifndef OBSERVATION_SPACES_H
#define OBSERVATION_SPACES_H
#include <torch/torch.h>
namespace octorl {

    struct StepReturn {
        //ObservationSpace observation;
        torch::Tensor observation;
        float reward = 0;
        bool terminated = false;
        bool goal = false;
        bool done = false;

        StepReturn() {};
        
        //print();
    };
    
    struct DiscreteActionSpace {
        int action_space_size;
        int *actions;

        DiscreteActionSpace() {};
        DiscreteActionSpace(int act_size);
        ~DiscreteActionSpace();
    };

    class ObservationSpace {
        public:
            ObservationSpace(int size, int dim);

        private:
        protected:


    };
} 
#endif