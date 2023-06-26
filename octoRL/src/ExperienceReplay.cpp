#include "../include/ExperienceReplay.hpp"

octorl::Memory::Memory(int s, torch::Tensor st, int  at, float rt, torch::Tensor st1, bool d) {
    step = s;
    state = st.clone();
    action = at;
    reward = rt;
    next_state = st1.clone();
    done = d;
}


octorl::ExperienceReplay::ExperienceReplay(size_t size) {
    max_size = size;
}

std::deque<octorl::Memory> octorl::ExperienceReplay::sample(int batch_size) {
    
    std::deque<octorl::Memory> sample_set;
    auto gen = std::mt19937 {std::random_device {}()};

    std::sample(replay_buffer.begin(), replay_buffer.end(),std::back_inserter(sample_set), batch_size, gen);

    return sample_set;
}


// returns true if buffer is full
bool octorl::ExperienceReplay::addToReplayBuffer(octorl::Memory e) {
    if(replay_buffer.size() == max_size) {
        replay_buffer.pop_front();
        replay_buffer.push_back(e);
        return true; 
    }   
    replay_buffer.push_back(e);
    return false;
}

int octorl::ExperienceReplay::getSize() {
    return replay_buffer.size();
}