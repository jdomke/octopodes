#include "../../include/agents/A3C.hpp"


//https://github.com/dennybritz/reinforcement-learning/blob/master/PolicyGradient/a3c/train.py





// make sure to check that number of eps is greater then num workers
octorl::A3C::A3C(std::shared_ptr<octorl::EnvironmentsBase> environment, size_t buffer_size, octorl::Mlp policy_model,octorl::Mlp actor_model,
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
    loadstatedict(critic, policy_model);
    loadstatedict(actor, actor_model);
    critic_optimizer = std::make_shared<torch::optim::Adam>(critic.parameters(), lr);
    actor_optimizer = std::make_shared<torch::optim::Adam>(actor.parameters(), lr);
    actor_optimizer->zero_grad();
    critic_optimizer->zero_grad();
    std::vector<torch::Tensor> actor_gradients = actor.emptyGradientHolder();
    std::vector<torch::Tensor> critic_gradients = critic.emptyGradientHolder();
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


        //torch::Tensor actor_loss = -1*torch::log(torch::matmul(prob,actor_memory[i].action_mask) + 1e-10)*(R - actor_memory[i].value)/scale;
        torch::Tensor actor_loss = -1*torch::log(prob[0][actor_memory[i].action] + 1e-10)*(R - actor_memory[i].value)/scale;
        torch::Tensor entropy = torch::sum(prob*torch::log(prob * 1e-10));
        //std::cout<<entropy<<std::endl;
        actor_loss += entropy_param*entropy;
        torch::Tensor value = critic.forward(actor_memory[i].state).to(device);
        torch::Tensor value_loss = torch::mse_loss(R, value)/(scale);  
        
        
        actor_loss.backward();
        value_loss.backward();
    }
    //std::cout<<actor.parameters()[actor.parameters().size()-1].grad()<<std::endl;
    //std::cout<<critic.parameters()[critic.parameters().size()-1].grad()<<std::endl;

    sendGradientSrc();
    sendActorGradient(actor_memory.size());
    sendCriticGradient(actor_memory.size());    
    recvActorModel();
    recvCriticModel();
    actor_memory.clear();
    memory.clear();
}
/*
bool octorl::A3C::workerRun() {
    auto init_obs = env->reset().observation;
    auto act = action(init_obs);
    auto obs = env->step(act);
    //addToLocalMemory(init_obs, act, obs.reward, obs.observation, obs.done);
       actor_memory.push_back(octorl::ActorMemory(init_obs, act, env->getActionSize(), obs.reward + 
        gamma*critic.forward(init_obs) - critic.forward(init_obs)));
    else
        actor_me    if(!obs.done)
 mory.push_back(octorl::ActorMemory(init_obs, act, env->getActionSize(),obs.reward - critic.forward(init_obs)));//torch::tensor({obs.reward})));

    memory.push_back(std::make_pair(octorl::Memory(-1,init_obs, act, obs.reward, obs.observation, obs.done),obs.reward + 
        gamma*critic.forward(init_obs)[0].item().toDouble()));
        //gamma*critic.forward(stateActionPair(obs.observation, act))[0].item().toDouble()));
    int steps = 1;
    int running_reward = obs.reward;
    while(!obs.done) {
        init_obs = obs.observation;
        act = action(init_obs);
        obs = env->step(act);
        
        if(!obs.done)
            actor_memory.push_back(octorl::ActorMemory(init_obs, act, env->getActionSize(), obs.reward + 
            gamma*critic.forward(init_obs) - critic.forward(init_obs)));
        else
            actor_memory.push_back(octorl::ActorMemory(init_obs, act, env->getActionSize(), obs.reward -critic.forward(init_obs)));//torch::tensor({obs.reward})));
        
        if(!obs.done)
            memory.push_back(std::make_pair(octorl::Memory(-1,init_obs, act, obs.reward, obs.observation, obs.done),obs.reward + 
            gamma*critic.forward(init_obs)[0].item().toDouble()));
            //gamma*critic.forward(stateActionPair(obs.observation, act))[0].item().toDouble()));
        else
            memory.push_back(std::make_pair(octorl::Memory(-1,init_obs, act, obs.reward, obs.observation, obs.done),obs.reward));
        
        running_reward += obs.reward;
        steps++;
    }
    calculateGradient();
    
    return recvKeepRunning();
}*/


/*
// can alternatively be used to do multiple batches per gradient send
void octorl::A3C::calculateGradient() {
    actor_optimizer->zero_grad();
    critic_optimizer->zero_grad();        
    
    std::vector<torch::Tensor> obs_vec;
    //std::vector<torch::Tensor> advantage;
    torch::Tensor q_val = torch::zeros({(int)memory.size()}).to(device);
    torch::Tensor advantage = torch::zeros({(int)memory.size()}).to(device);
   // torch::Tensor advantage = torch::zeros({(int)memory.size()}).to(device);
     int i;
    for(i = 0; i < actor_memory.size(); i++)
    { 
        torch::Tensor prob = actor.forward(actor_memory[i].state).to(device);
        //std::cout<<prob<<std::endl;
        //std::cout<<torch::matmul(prob,actor_memory[i].action_mask)<<std::endl;
        torch::Tensor actor_loss = -1*torch::log(torch::matmul(prob,actor_memory[i].action_mask))*actor_memory[i].advantage;
        //std::cout<<actor_loss<<std::endl;
        actor_loss.backward();
        obs_vec.push_back(memory[i].first.state);
        q_val[i] = memory[i].second;
        //mask[i] = actor_memory[i].action_mask;

        advantage[i] = actor_memory[i].advantage.item().toDouble();;
    }
    //std::cout<<"past actor\n";

//    std::cout<<i<<std::endl;
 //   std::cout<<actor.parameters()[0].grad()<<std::endl;
     torch::TensorList input {obs_vec};
    torch::Tensor input_batch = torch::cat(input).to(device);
    
    //torch::TensorList a {advantage};
    //torch::Tensor adv_batch = torch::cat(a).to(device);
   // std::cout<<torch::matmul(actor.forward(input_batch),mask)<<std::endl;
    
    torch::Tensor value = critic.forward(input_batch).reshape({input_batch.size(0)});

    torch::Tensor value_loss = torch::mse_loss(q_val, value);
    value_loss.backward();
    sendGradientSrc();
    sendActorGradient();
    sendCriticGradient();    
    recvActorModel();
    recvCriticModel();
    actor_memory.clear();
    memory.clear();
}*/

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
   // std::cout<<"grad:"<<actor.parameters()[0].grad()<<std::endl;
   // std::cout<<"vale:"<<actor.parameters()[0]<<std::endl;
    int num_steps = (int) buffer[0];
    actor.applyGradient(buffer++, num_steps);
   // std::cout<<"grad:"<<actor.parameters()[0].grad()<<std::endl;
    actor_optimizer->step();
   // std::cout<<"vale:"<<actor.parameters()[0]<<std::endl;
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

