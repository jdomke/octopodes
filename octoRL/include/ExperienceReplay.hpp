#ifndef EXPERIENCE_REPLAY_H
#define EXPERIENCE_REPLAY_H
#include <torch/torch.h>
#include <vector>
#include <utility>
#include <algorithm>
#include <iterator>
#include <random>


// experience replay for discrete action spaces
namespace octorl {

    struct Memory {
        int step;
        torch::Tensor state;
        torch::Tensor next_state;
        int action;
        float reward;
        bool done;
        Memory(int s, torch::Tensor st, int at, float rt, torch::Tensor st1, bool d);
    };
    
    struct ActorMemory {
        torch::Tensor state;
        torch::Tensor action_mask;
        torch::Tensor advantage;
        torch::Tensor value;
        int action;
        float reward;
        bool done;
        ActorMemory(torch::Tensor obs, torch::Tensor v, int act, int action_size, float r, bool d);
    };

    class ExperienceReplay {
        private:
            size_t max_size;
            std::deque<Memory> replay_buffer;

        public:
            ExperienceReplay() {};
            ExperienceReplay(size_t size);
            std::deque<Memory> sample(int batch_size);
            bool addToReplayBuffer(Memory e);
            int getSize();
    };

}

#endif