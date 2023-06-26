#ifndef DQN_ASYNC_H
#define DQN_ASYNC_H

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
#include "../Mlp.hpp"
#include "mpi.h"



namespace octorl {

    enum tags {
        model_tag,
        batch_tag,
        keep_running_tag
    };

    struct actor {
        std::shared_ptr<EnvironmentsBase> env;
        Mlp model;
        Mlp target;
    };

    class DqnAsync {

        public: 
            DqnAsync(std::shared_ptr<EnvironmentsBase> environment, size_t buffer_size, Mlp policy_model, std::vector<std::shared_ptr<EnvironmentsBase>> act_envs,
                float g, float eps, float decay, float eps_min,int ep_count, int seed, double lr,int batch,int argc, char** argv);      
            void run();
            std::vector<actor> actors;
            bool sendModel(int r);
            bool recvModel();
            int action(torch::Tensor state);
            bool sendBatch(int done);
            std::pair<int,int> recvBatch();
            void test();
            int recvBatchAndTrain();
            void sendKeepRunning(bool run, int dst);
            bool recvKeepRunning();
            void pushToBuffer(Memory m);
            float trainOnBatch(std::vector<Memory> batch);
            void updateTarget();
            torch::Tensor calcTargetF(Memory m);

        private:
            std::default_random_engine gen;
            std::uniform_real_distribution<float> distribution{std::uniform_real_distribution<float>(0,1)};
            std::shared_ptr<EnvironmentsBase> env;
            std::vector<std::shared_ptr<EnvironmentsBase>> actor_envs;
            octorl::Mlp model;
            octorl::Mlp target_model;
            ExperienceReplay exp_buffer;
            std::deque<std::shared_ptr<float>> local_memory;
            std::shared_ptr<torch::optim::Adam> model_optimizer;
            std::shared_ptr<torch::optim::Adam> target_optimizer;
            double learning_rate; 
            float gamma;
            float epsilon;
            float epsilon_decay;
            float epsilon_min;
            int batch_freq;
            int episodes;
            int batch_size;
            int epochs;
            int local_size;
            int local_batch;
            int rank;
            int numranks;
            int steps;
            int updateCount;
            void addToLocalMemory(torch::Tensor init_obs, int act, float reward, torch::Tensor next_obs, int done);
            MPI_Status status;
            MPI_Request req1, req2; 
    };
}

#endif