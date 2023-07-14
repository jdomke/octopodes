#ifndef A2C_H
#define A2C_H

#include <torch/torch.h>
#include <random>
#include <cstdlib>
#include <utility>
#include <list>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cmath>
#include "../ExperienceReplay.hpp"
#include "../EnvironmentsBase.hpp"
#include "../helper.hpp"
//#include "../Mlp.hpp"
#include "../Policy.hpp"
#include "../MpiTags.hpp"
#include "mpi.h"

namespace octorl {

    class A2C {

        public:
            A2C(std::shared_ptr<EnvironmentsBase> environment, size_t buffer_size, Policy policy_model, Policy actor_model,
               float g, int ep_count, int seed, double lr,int batch, int r, int nr);

            void run();
            void test();
            void learnerRun();
            bool workerRun();
            bool sendCriticModel(int r);
            bool sendActorModel(int r);
            bool broadcastActorModel();
            bool broadcastCriticModel();
            bool recvBroadcastActorModel();
            bool recvBroadcastCriticModel();
            bool recvCriticModel();
            bool recvActorModel();
            void sendKeepRunning(bool run, int dst);
            void broadcastKeepRunning(bool run);
            bool recvBroadcastKeepRunning();
            bool recvKeepRunning();
            void recvBatch();
            void sendBatch();
            void calculateQValAddLocal();
            void train();
            int action(torch::Tensor state);


        private:
            std::default_random_engine gen;
            std::uniform_real_distribution<float> distribution{std::uniform_real_distribution<float>(0,1)};
            std::shared_ptr<EnvironmentsBase> env;
            Policy critic;
            Policy actor;

            torch::Device device =torch::kCPU;
            std::shared_ptr<torch::optim::Adam> critic_optimizer;
            std::shared_ptr<torch::optim::Adam> actor_optimizer;
            std::deque<std::shared_ptr<float>> local_memory;
            std::vector<ActorMemory> memory;
            std::vector<std::pair<ActorMemory, float>> batch_memory;
            double learning_rate; 
            float gamma;
            int batch_freq;
            int episodes;
            int batch_size;
            int local_batch_size;
            int epochs;
            int local_size;
            int local_memory_size;
            int local_batch;
            int entropy_param;
            int rank;
            int num_ranks;
            int steps;
            int update_count;
            void addToLocalMemory(torch::Tensor init_obs, int act, float reward, float R, int done);


    };
}

#endif