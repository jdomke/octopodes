#ifndef A3C_H
#define A3C_H

#include <torch/torch.h>
#include <random>
#include <cstdlib>
#include <utility>
#include <list>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cmath>
#include "../helper.hpp"
#include "../ExperienceReplay.hpp"
#include "../EnvironmentsBase.hpp"
#include "../Mlp.hpp"
#include "../Policy.hpp"
#include "../MpiTags.hpp"
#include "mpi.h"


namespace octorl {

    class A3C {
        

        public: 
            A3C(std::shared_ptr<EnvironmentsBase> environment, size_t buffer_size, Policy policy_model, Policy actor_model,
               float g, int ep_count, int seed, double lr,int batch, int r, int nr);
            void run();
            void test();
            void globalNetworkRun();
            bool workerRun();
            bool sendCriticModel(int r);
            bool sendActorModel(int r);
            bool recvCriticModel();
            bool recvActorModel();
            void sendKeepRunning(bool run, int dst);
            bool recvKeepRunning();
            int action(torch::Tensor state);
            void calculateGradient(torch::Tensor R);
            void calculateActorGradient();
            void sendGradientSrc();
            int recvGradientSrc();
            void sendActorGradient(int mem_size);
            void sendCriticGradient(int mem_size);
            void recvActorGradientAndStep(int src);
            void recvCriticGradientAndStep(int src);
            torch::Tensor stateActionPair(torch::Tensor state, int act);


        private:
            std::default_random_engine gen;
            std::uniform_real_distribution<float> distribution{std::uniform_real_distribution<float>(0,1)};
            std::shared_ptr<EnvironmentsBase> env;
            Policy critic;
            Policy actor;

            torch::Device device =torch::kCPU;
            std::shared_ptr<torch::optim::Adam> critic_optimizer;
            std::shared_ptr<torch::optim::Adam> actor_optimizer;
            std::vector<std::pair<Memory, float>> memory;
            std::vector<ActorMemory> actor_memory;
            double learning_rate; 
            float gamma;
            int batch_freq;
            int episodes;
            int batch_size;
            int epochs;
            int local_size;
            int local_batch;
            int entropy_param;
            int rank;
            int num_ranks;
            int local_memory_size;
            int steps;
            int update_count;
            int t_max;
            int t;
    	    int time_steps;
            int t_start;
    };
}





#endif
