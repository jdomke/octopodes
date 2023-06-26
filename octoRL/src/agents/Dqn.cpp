#include "../../include/agents/Dqn.hpp"
#include <iostream>


octorl::Dqn::Dqn(std::shared_ptr<octorl::EnvironmentsBase> environment, size_t buffer_size,octorl::Mlp policy_model, 
                float g, float eps, float decay, float eps_min, int ep_count, int seed, double lr, int batch) {

    env = environment;
    exp_buffer = ExperienceReplay(buffer_size);
    gamma = g;
    epsilon = eps;
    epsilon_decay = decay;
    epsilon_min = eps_min;
    episodes = ep_count;
    model = policy_model;
    target_model = policy_model;
    loadstatedict(target_model,policy_model);
    loadstatedict(model,policy_model);
    learning_rate = lr;
    batch_size = batch;
    model_optimizer = std::make_shared<torch::optim::Adam>(model.parameters(), lr);
    target_optimizer = std::make_shared<torch::optim::Adam>(target_model.parameters(), lr);
    srand(seed);
}

std::pair<int,int> octorl::Dqn::action(torch::Tensor state) {
    int act;
    float r = distribution(gen);
    if(r < epsilon) {
        //updateEpsilon();
        int r = rand() % env->getActionSize();
        return std::make_pair(r,0);
    }
    act = model.forward(state).argmax().item().toInt();
    
    return std::make_pair(act,1);
}

void octorl::Dqn::updateEpsilon() {
    if(epsilon > epsilon_min)
        epsilon *= epsilon_decay;
    else
        epsilon = epsilon_min;
}

torch::Tensor octorl::Dqn::calcTargetF(octorl::Memory m) {
    double expected_q = 0;
    if(!m.done) {
        double amax = target_model.forward(m.next_state).amax().item().toDouble();
        expected_q = gamma * amax;
    }

    double target = m.reward + expected_q;

    torch::Tensor target_f = target_model.forward(m.state);
    
    target_f[0][m.action] = target;
    
    return target_f;
}

std::deque<octorl::Memory> octorl::Dqn::sample() {
    int size = batch_size;
    if(exp_buffer.getSize() < batch_size)
        size = exp_buffer.getSize();
    
    return exp_buffer.sample(size);
}

void octorl::Dqn::printModelParameters() {
    for(auto x : model.parameters()){
        std::cout<<x<<std::endl;
    }
}
void octorl::Dqn::printTargetModelParameters() {
    for(auto x : target_model.parameters()){
        std::cout<<x<<std::endl;
    }
}
void octorl::Dqn::update_target()  {
    //target_model = model;
    loadstatedict(target_model,model);

}

double octorl::Dqn::train() {
    std::vector<torch::Tensor> in_vec;
    std::vector<torch::Tensor> out_vec;

    for(auto i : sample()){
        
        in_vec.push_back(i.state);
        out_vec.push_back(calcTargetF(i));
    }
    torch::TensorList input {in_vec};
    torch::TensorList target {out_vec};
    torch::Tensor input_batch = torch::cat(input);
    torch::Tensor target_batch = torch::cat(target);
    model_optimizer->zero_grad();
    //uto input_batch = torch::data::datasets::BatchDataset(in_vec);
    //auto target = torch::data::datasets::TensorDataset(out_vec);
    torch::Tensor output = model.forward({input_batch});
    torch::Tensor loss = torch::mse_loss(target_batch,output);

    loss.backward();

    model_optimizer->step();

    //std::cout<<target_batch - output<<std::endl;
    //updateEpsilon(); 
    return loss.item().toDouble();

}

void octorl::Dqn::pushToBuffer(octorl::Memory m) {
    // add check for when buffer is full
    exp_buffer.addToReplayBuffer(m);
}

int octorl::Dqn::getBufferSize() {
    return exp_buffer.getSize();
}

int octorl::Dqn::getBatchSize() {
    return batch_size;
}

torch::Tensor octorl::Dqn::modelPredict(torch::Tensor x) {
    return model.forward(x);
}
torch::Tensor octorl::Dqn::targetPredict(torch::Tensor x) {
    return model.forward(x);
}

bool octorl::Dqn::modelsMatch() {
    auto x = model.parameters();
    auto y = target_model.parameters();
    std::cout<<x.size()<<std::endl;
    //std::cout<<x[0] == y[0]<<std::endl;

    for(int i = 0; i < x.size();i++)
        std::cout<<(x[i] == y[i])<<std::endl;

    return true;
}   