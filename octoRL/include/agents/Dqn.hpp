#ifndef DQN_H
#define DQN_H

#include <torch/torch.h>
#include <random>
#include <cstdlib>
#include <utility>
#include "../ExperienceReplay.hpp"
#include "../EnvironmentsBase.hpp"
#include "../Mlp.hpp"

//#include "../AgentBase.hpp"
// will abstract later

namespace octorl {

    class Dqn {
        public:
            Dqn(std::shared_ptr<EnvironmentsBase> environment, size_t buffer_size, Mlp policy_model, 
                float g, float eps, float decay, float eps_min,int ep_count, int seed, double lr,int batch);
            torch::Tensor modelPredict(torch::Tensor x);
            torch::Tensor targetPredict(torch::Tensor x);
            std::pair<int,int> action(torch::Tensor state);
            torch::Tensor calcTargetF(Memory m);
            std::deque<Memory> sample();
            void printModelParameters(); 
            void printTargetModelParameters(); 
            void update_target();
            double train();
            void pushToBuffer(Memory m);
            int getBufferSize();
            int getBatchSize();
            void run();
            void updateEpsilon();
            bool modelsMatch();
            
        private:

            std::default_random_engine gen;
            std::uniform_real_distribution<float> distribution{std::uniform_real_distribution<float>(0,1)};
            std::shared_ptr<EnvironmentsBase> env;
            octorl::Mlp model;
            octorl::Mlp target_model;
            ExperienceReplay exp_buffer;
            std::shared_ptr<torch::optim::Adam> model_optimizer;
            std::shared_ptr<torch::optim::Adam> target_optimizer;
            double learning_rate; 
            float gamma;
            float epsilon;
            float epsilon_decay;
            float epsilon_min;
            int episodes;
            int batch_size;
            

    };

}

#endif