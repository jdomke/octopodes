#include <map>
#include <list>
#include <torch/torch.h>
#include "mpi.h"
#include "EnvironmentsBase.hpp"
#include "Mlp.hpp"

namespace octorl {
    std::map<at::ScalarType, MPI_Datatype> mpiDatatype = {
    {at::kByte, MPI_UNSIGNED_CHAR},
    {at::kChar, MPI_CHAR},
    {at::kDouble, MPI_DOUBLE},
    {at::kFloat, MPI_FLOAT},
    {at::kInt, MPI_INT},
    {at::kLong, MPI_LONG},
    {at::kShort, MPI_SHORT},
    };
    
    class AsyncLearner {
        private:
            bool is_actor;
            bool is_learner;
            int batch_frequency;
            int rank;
            int steps;
            int done;
            int model_count;
            int total_reward;
            int episode_count;
            int total_episodes;
            int update_target_frequency;
            std::list<float> episode_reward_list;

        public:
            AsyncLearner();
            AsyncLearner();




    };
}