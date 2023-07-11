#include "../../include/agents/A3C.hpp"


//https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/a3c/train.py





// make sure to check that number of eps is greater then num workers
octorl::A3C::A3C(std::shared_ptr<octorl::EnvironmentsBase> environment, size_t buffer_size, octorl::Policy policy_model,octorl::Policy actor_model,
               float g, int ep_count, int seed, double lr,int batch, int r, int nr) {

    env = environment;
    env->reset();
    gamma = g; 
    learning_rate = lr;
    episodes = ep_count;
    batch_size = batch;
    rank = r;
    num_ranks = nr;
    local_size = buffer_size;   
    t_max = batch;
    time_steps = ep_count;
    t = 0;
    entropy_param = 0.0001;
    if (torch::cuda::is_available()) {                                                                                                                                                                                     std::cout << "CUDA is available! Training on GPU." << std::endl;                                                                                                                                           
        device = torch::kCUDA;                                                                                                                                                                                     
    }   

    critic = policy_model;
    actor = actor_model;
    octorl::loadstatedict2(critic, policy_model);
    octorl::loadstatedict2(actor, actor_model);
    critic_optimizer = std::make_shared<torch::optim::Adam>(critic.parameters(), lr);
    actor_optimizer = std::make_shared<torch::optim::Adam>(actor.parameters(), lr);
    actor_optimizer->zero_grad();
    critic_optimizer->zero_grad();
    srand(seed);

}

void octorl::A3C::test() {
    std::cout<<"test on rank: "<<rank<<std::endl;
    int test_goals = 0;
    float avg_reward= 0;
    torch::Tensor init_obs;
    for(int i = 0; i < 100; i++){
        float rewards = 0; 
        auto obs = env->reset();    
        rewards += obs.reward;
        avg_reward += obs.reward;
        int act;
        while(!obs.done) {
    	    init_obs = obs.observation;
	    act = action(obs.observation);
            obs = env->step(act);
            rewards += obs.reward;
            avg_reward += obs.reward;
	}
        if(obs.goal){
            test_goals++;
            //std::cout<<rewards<<std::endl;
        }
        
    }

    std::cout<<"test Goals: "<<test_goals<<std::endl;
    std::cout<<"Avg Reward: "<<avg_reward/100<<std::endl;
}

void octorl::A3C::run() {

    if(rank == 0) {
        globalNetworkRun();
        test();
    }
    else {
        recvActorModel();
        recvCriticModel();
        while(workerRun()){}
        std::cout<<rank<<" done running\n";
    }
}


void octorl::A3C::globalNetworkRun() {
    // initalize workers
    for(int i =1; i < num_ranks; i++) {
        sendActorModel(i);
        sendCriticModel(i);
        //sendKeepRunning(true,i);
    }
    int src;
    for(int i = 0; i < time_steps; i++) {
        // MESSAGE ORDER IS VERY IMPORTANT
        if(i % 100 == 0) {
            std::cout<<"Ep: "<<i<<std::endl;
            test();
        }
        src = recvGradientSrc();
        recvActorGradientAndStep(src);
        recvCriticGradientAndStep(src);
        sendActorModel(src);
        sendCriticModel(src);
        if(time_steps - i <= num_ranks - 1) {
            sendKeepRunning(false, src);
        }
        else
            sendKeepRunning(true, src);
    }
}


bool octorl::A3C::workerRun() {
    t_start = t;
    auto init_obs = env->getState();
   

    auto act = action(init_obs);
    auto obs = env->step(act);

    actor_memory.push_back(octorl::ActorMemory(init_obs, critic.forward(init_obs).to(device).detach(), act, env->getActionSize(), obs.reward, obs.done));
    t++;
    int steps = 1;
     while(!obs.done) {
        init_obs = obs.observation;
        act = action(init_obs);
        obs = env->step(act);
        actor_memory.push_back(octorl::ActorMemory(init_obs, critic.forward(init_obs).to(device).detach(), act, env->getActionSize(), obs.reward, obs.done));
        t++;

        if(t - t_start == t_max && !obs.done) {
            calculateGradient(critic.forward(init_obs).to(device).detach());
            return recvKeepRunning();

        }
    }
    calculateGradient(torch::zeros(1).to(device).detach());
    t = 0;
    env->reset();

    return recvKeepRunning();
}

void octorl::A3C::calculateGradient(torch::Tensor R) {
    actor_optimizer->zero_grad();

    critic_optimizer->zero_grad();  
    
    torch::Tensor scale = torch::tensor({(int)actor_memory.size()}).to(device);
    for(int i = actor_memory.size() - 1; i >= 0; i--) {
        R = actor_memory[i].reward + gamma * R; 
        torch::Tensor prob = actor.forward(actor_memory[i].state).to(device);
        
        
        torch::Tensor actor_loss = -1*torch::log(torch::matmul(prob,actor_memory[i].action_mask) + 1e-10)*(R - actor_memory[i].value);
        torch::Tensor entropy = torch::sum(prob*torch::log(prob * 1e-10));
        actor_loss += entropy_param*entropy;
        actor_loss /= scale;
        torch::Tensor value = critic.forward(actor_memory[i].state).requires_grad_(true).to(device);
        torch::Tensor value_loss = torch::mse_loss(R, value)/scale;  
        //std::cout<<actor_loss<<std::endl;
        actor_loss.backward();
        value_loss.backward();
    }

    sendGradientSrc();
    sendActorGradient(actor_memory.size());
    sendCriticGradient(actor_memory.size());    
    recvActorModel();
    recvCriticModel();
    actor_memory.clear();
    memory.clear();
}


torch::Tensor octorl::A3C::stateActionPair(torch::Tensor state, int act) {
    torch::Tensor input = torch::zeros({env->getObservationSize() + 1});
    
    for(int i = 0; i < env->getObservationSize(); i++)
        input[i] = state[0][i];
    input[env->getObservationSize()] = (float) act/env->getActionSize();
    return input;
}


void octorl::A3C::sendGradientSrc() {
    MPI_Send(&rank, 1, MPI_INT, 0, octorl::gradient_sync_tag, MPI_COMM_WORLD);
}

// can be done with mpi status as well
int octorl::A3C::recvGradientSrc() {
    int r;
    MPI_Recv(&r, 1, MPI_INT, MPI_ANY_SOURCE, octorl::gradient_sync_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return r;
}

// should move these gradient sends to model class
void octorl::A3C::sendActorGradient(int mem_size) {
    float *buffer = new float[actor.getElementCount() + 1];   
    int b = 0;
    buffer[b++] = (float) mem_size; 
    for(auto i : actor.parameters()){
        auto flat = torch::flatten(i.grad());
        for(int j = 0; j < flat.numel(); j++)
            buffer[b++] = flat[j].item<float>();   
    }
    MPI_Send(buffer, actor.getElementCount(), MPI_FLOAT, 0, octorl::gradient_tag, MPI_COMM_WORLD);
}

void octorl::A3C::sendCriticGradient(int mem_size) {
    float *buffer = new float[critic.getElementCount() + 1];   
    int b = 0; 
    buffer[b++] = (float) mem_size; 
    for(auto i : critic.parameters()){
        auto flat = torch::flatten(i.grad());
        for(int j = 0; j < flat.numel(); j++)
            buffer[b++] = flat[j].item<float>();   
    }
    MPI_Send(buffer, critic.getElementCount(), MPI_FLOAT, 0, octorl::gradient_tag, MPI_COMM_WORLD);
}

void octorl::A3C::recvActorGradientAndStep(int src) {
    float *buffer = new float[actor.getElementCount() + 1];   
    MPI_Status status;
    MPI_Recv(buffer, actor.getElementCount(), MPI_FLOAT, src, octorl::gradient_tag,MPI_COMM_WORLD, &status);
    int num_steps = (int) buffer[0];
    actor.applyGradient(buffer++, num_steps);
    actor_optimizer->step();
}

void octorl::A3C::recvCriticGradientAndStep(int src) {
    float *buffer = new float[critic.getElementCount() + 1];   
    MPI_Status status;

    MPI_Recv(buffer, critic.getElementCount(), MPI_FLOAT, src, octorl::gradient_tag,MPI_COMM_WORLD, &status);
    int num_steps = (int) buffer[0];
    critic.applyGradient(buffer++, num_steps);

    critic_optimizer->step();
}

void octorl::A3C::sendKeepRunning(bool run, int dst) {
    MPI_Send(&run, 1, MPI_C_BOOL, dst, octorl::keep_running_tag, MPI_COMM_WORLD);
}

bool octorl::A3C::recvKeepRunning() {
    bool kr;
    MPI_Recv(&kr, 1, MPI_C_BOOL, 0, octorl::keep_running_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return kr;
}


int octorl::A3C::action(torch::Tensor state) {
   
    //torch::Tensor out = actor.forward(state).to(device).contiguous();
    torch::Tensor out = actor.forward(state).contiguous();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> d(out.data_ptr<float>(), out.data_ptr<float>() + out.numel());
    
    return d(gen);
}

bool octorl::A3C::sendCriticModel(int r) {
    float *buffer = new float[critic.getElementCount()];    
    critic.serialize(buffer);
    MPI_Send(buffer, critic.getElementCount(), MPI_FLOAT, r, octorl::model_tag, MPI_COMM_WORLD);
    delete[] buffer;
    return true;
}

bool octorl::A3C::sendActorModel(int r) {
    float *buffer = new float[actor.getElementCount()];    
    actor.serialize(buffer);
    MPI_Send(buffer, actor.getElementCount(), MPI_FLOAT, r, octorl::model_tag, MPI_COMM_WORLD);

    delete[] buffer;
    return true;
}

bool octorl::A3C::recvCriticModel() {
    float *buffer = new float[critic.getElementCount()];    
    MPI_Recv(buffer, critic.getElementCount(), MPI_FLOAT, 0, octorl::model_tag, MPI_COMM_WORLD,
         MPI_STATUS_IGNORE);
    critic.loadFromSerial(buffer);
    delete[] buffer;
    return true;
}

bool octorl::A3C::recvActorModel() {
    float *buffer = new float[actor.getElementCount()];    
    MPI_Recv(buffer, actor.getElementCount(), MPI_FLOAT, 0, octorl::model_tag, MPI_COMM_WORLD,
         MPI_STATUS_IGNORE);
    actor.loadFromSerial(buffer);
    //std::cout<<actor.parameters()[actor.parameters().size()-1]<<std::endl;
    delete[] buffer;
    return true;
}

