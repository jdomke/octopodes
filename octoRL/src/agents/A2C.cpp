#include "../../include/agents/A2C.hpp"




octorl::A2C::A2C(std::shared_ptr<octorl::EnvironmentsBase> environment, size_t buffer_size, octorl::Policy policy_model,octorl::Policy actor_model,
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
    local_batch_size = std::ceil(batch/(nr-1));
    if (torch::cuda::is_available()) {   
        std::cout << "CUDA is available! Training on GPU." << std::endl;                                                                                                                                           
    //    device = torch::kCUDA;                                                                                                                                                                                     
    }   
    //local_memory_size = 400;
    local_memory_size = local_batch_size;
    
    entropy_param = 0.0001;
    critic = policy_model;
    actor = actor_model;
    critic.to(device);
    actor.to(device);
    octorl::loadstatedict2(critic, policy_model);
    octorl::loadstatedict2(actor, actor_model);
    

    critic_optimizer = std::make_shared<torch::optim::Adam>(critic.parameters(), lr);
    actor_optimizer = std::make_shared<torch::optim::Adam>(actor.parameters(), lr);
    actor_optimizer->zero_grad();
    critic_optimizer->zero_grad();
    rand_seed = seed;
    srand(seed);
    gen.seed(rand_seed);
    sample_gen.seed(rand_seed);

}

void octorl::A2C::test() {
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
	    //init_obs = obs.observation;
	    act = action(obs.observation.to(device));
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

void octorl::A2C::run() {
    if(rank == 0) {
        MPI_Barrier(MPI_COMM_WORLD); 
        broadcastActorModel();
        broadcastCriticModel();
        learnerRun();
        test();
    }
    else {
        MPI_Barrier(MPI_COMM_WORLD); 
        recvBroadcastActorModel();
        recvBroadcastCriticModel();
        // recvActorModel();
        // recvCriticModel();
        while(workerRun()){}
        std::cout<<rank<<" done running\n";
    }
}

void octorl::A2C::learnerRun() {
    /*for(int i =1; i < num_ranks; i++) {
        sendActorModel(i);
        sendCriticModel(i);
    }*/
    
    for(int e = 0; e < episodes; e++) {
        for(int i = 1; i < num_ranks; i++)
            recvBatch();
        train();
        if(e % 100 == 0) {
            std::cout<<"Epoch: "<<e<<std::endl;
            test();
        }
        MPI_Barrier(MPI_COMM_WORLD); 
        broadcastActorModel();
        broadcastCriticModel();
        if(e+1 < episodes)
            broadcastKeepRunning(true);
        else
            broadcastKeepRunning(false);
        // for(int i = 1; i < num_ranks; i++) {
        //     sendActorModel(i);
        //     sendCriticModel(i);
        //     if(e+1 < episodes)
        //         sendKeepRunning(true,i);
        //     else
        //         sendKeepRunning(false,i);
        // }

    }
}

bool octorl::A2C::workerRun() {
   // std::cout<<"worker run\n";
    auto init_obs = env->reset().observation.to(device);
    auto act = action(init_obs);
    //std::cout<<"Acted\n";
    auto obs = env->step(act);
    memory.push_back(octorl::ActorMemory(init_obs, critic.forward(init_obs).to(device).detach(), act, env->getActionSize(), obs.reward, obs.done));
    int i = 1;
    while(!obs.done) {
      //  std::cout<<"siomming\n";
	    obs.observation.to(device);
        init_obs = obs.observation;
        act = action(init_obs);
        obs = env->step(act);
        //addToLocalMemory(init_obs, act, obs.reward, obs.done);
       //std::cout<<i++<<std::endl;
        memory.push_back(octorl::ActorMemory(init_obs, critic.forward(init_obs).to(device).detach(), act, env->getActionSize(), obs.reward, obs.done));
    
       // std::cout<<memory.size()<<std::endl;
       // std::cout<<"Goal: "<<obs.goal<<std::endl;
    }
    //std::cout<<"post sim\n";    
    calculateQValAddLocal();

   // std::cout<<memory.size()<<std::endl;
    //std::cout<<local_memory.size()<<std::endl;
    // need to calculate q_val
    if(local_memory.size() >= local_memory_size) {
        sendBatch();
        MPI_Barrier(MPI_COMM_WORLD); 
        recvBroadcastActorModel();
        recvBroadcastCriticModel();
        // recvActorModel();
        // recvCriticModel();
        memory.clear();
        local_memory.clear();
    }
    else // keep running till enough for a local batch
        return true;
    

    return recvBroadcastKeepRunning();
    
}

void octorl::A2C::calculateQValAddLocal() {
    float R = 0;

    for(int i = memory.size() - 1; i >= 0; i--) {
        R = memory[i].reward + gamma * R; 
        addToLocalMemory(memory[i].state, memory[i].action, memory[i].reward, R, memory[i].done);
    }
}

void octorl::A2C::train() {
    actor_optimizer->zero_grad();
    critic_optimizer->zero_grad();        
    
    std::vector<torch::Tensor> obs_vec;
    torch::Tensor q_val = torch::zeros({(int)batch_memory.size()}).to(device);
    torch::Tensor advantage = torch::zeros({(int)batch_memory.size()}).to(device);
    torch::Tensor mask = torch::zeros({(int)batch_memory.size(),env->getActionSize()}).to(device);
    for(int i = 0; i < batch_memory.size(); i++) {
        obs_vec.push_back(batch_memory[i].first.state);
        q_val[i] = batch_memory[i].second;
	
        advantage[i] = torch::sub(q_val[i],batch_memory[i].first.value).item().toDouble();

        mask[i] = batch_memory[i].first.action_mask;
    }
    torch::TensorList input {obs_vec};
    torch::Tensor input_batch = torch::cat(input).to(device);
    auto prob = actor.forward(input_batch);
    auto entropy = torch::mean(torch::sum(prob*torch::log(prob + 1e-10), -1)).to(device);
    //std::cout<<entropy<<std::endl;
    //std::cout<<input_batch<<std::endl;
    auto actor_loss = torch::mean(-1*torch::log(torch::sum(prob*mask,1) + 1e-10)*advantage);
    actor_loss += entropy_param*entropy;
    //std::cout<<actor_loss<<std::endl;
    
    actor_loss.backward();
    auto value = critic.forward({input_batch}).reshape((-1,q_val.size(0))).to(device);
    auto critic_loss = torch::mse_loss(q_val, value);
    critic_loss.backward();

    actor_optimizer->step();
    critic_optimizer->step();
    batch_memory.clear();
}

int octorl::A2C::action(torch::Tensor state) {
   
    //torch::Tensor out = actor.forward(state).to(device).contiguous();
    torch::Tensor out = actor.forward(state).contiguous();

    //std::random_device rd;
    //std::mt19937 gen(rand_seed);
    std::discrete_distribution<> d(out.data_ptr<float>(), out.data_ptr<float>() + out.numel());
    
    return d(gen);
}


void octorl::A2C::recvBatch() {
    float *batch = new float[local_batch_size * (env->getObservationSize() + 4)];
    MPI_Recv(batch,local_batch_size * (env->getObservationSize() + 4), MPI_FLOAT, MPI_ANY_SOURCE, octorl::batch_tag, MPI_COMM_WORLD,
    MPI_STATUS_IGNORE);


    int act, done;
    float reward, R;
    int b = 0;

    for(int i = 0; i < local_batch_size; i++) {
        torch::Tensor init_obs = torch::ones((env->getObservationSize()));
        //init_obs = init_obs.reshape({1,init_obs.size(0)});
        for(int j = 0; j < env->getObservationSize(); j++){
            init_obs[j] = batch[b++];
        }
        init_obs = env->shapeObservation(init_obs);
        
        act = (int) batch[b++];
        reward = batch[b++];
        R = batch[b++];
        done = (int) batch[b++];
        batch_memory.push_back(std::make_pair(octorl::ActorMemory(init_obs, critic.forward(init_obs).to(device).detach(), act, env->getActionSize(), reward, done),R));
    }
    delete[] batch;
}

void octorl::A2C::sendBatch() {
    float *batch = new float[local_batch_size * (env->getObservationSize() + 4)];
    int b = 0;

    std::deque<std::shared_ptr<float>> sample_set;
    //auto gen1 = std::mt19937 {std::random_device {}()};
    std::sample(local_memory.begin(), local_memory.end(),std::back_inserter(sample_set), local_batch_size, sample_gen);
    
    for(int i = 0; i < sample_set.size(); i++) {
        for(int j = 0; j < env->getObservationSize() + 4; j++){
            batch[b++] = sample_set[i].get()[j];
        }
    }
    MPI_Send(batch, local_batch_size * (env->getObservationSize() + 4), MPI_FLOAT, 0, octorl::batch_tag, MPI_COMM_WORLD);
    delete[] batch;
}

bool octorl::A2C::sendCriticModel(int r) {
    float *buffer = new float[critic.getElementCount()];    
    critic.serialize(buffer);
    MPI_Send(buffer, critic.getElementCount(), MPI_FLOAT, r, octorl::model_tag, MPI_COMM_WORLD);
    delete[] buffer;
    return true;
}

bool octorl::A2C::sendActorModel(int r) {
    float *buffer = new float[actor.getElementCount()];    
    actor.serialize(buffer);
    MPI_Send(buffer, actor.getElementCount(), MPI_FLOAT, r,  octorl::model_tag, MPI_COMM_WORLD);

    delete[] buffer;
    return true;
}

bool octorl::A2C::broadcastCriticModel() {
    float *buffer = new float[critic.getElementCount()];    
    critic.serialize(buffer);
    MPI_Bcast(buffer, critic.getElementCount(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    delete[] buffer;
    return true;
}
bool octorl::A2C::broadcastActorModel() {
    float *buffer = new float[actor.getElementCount()];    
    actor.serialize(buffer);
    MPI_Bcast(buffer, actor.getElementCount(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    delete[] buffer;
    return true;
}

bool octorl::A2C::recvBroadcastCriticModel() {
    float *buffer = new float[critic.getElementCount()];    
    MPI_Bcast(buffer, critic.getElementCount(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    critic.loadFromSerial(buffer);
    delete[] buffer;
    return true;
}

bool octorl::A2C::recvBroadcastActorModel() {
    float *buffer = new float[actor.getElementCount()];    
    MPI_Bcast(buffer, actor.getElementCount(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    actor.loadFromSerial(buffer);
    delete[] buffer;
    return true;
}

bool octorl::A2C::recvCriticModel() {
    float *buffer = new float[critic.getElementCount()];    
    MPI_Recv(buffer, critic.getElementCount(), MPI_FLOAT, 0, octorl::model_tag, MPI_COMM_WORLD,
         MPI_STATUS_IGNORE);
    critic.loadFromSerial(buffer);
    delete[] buffer;
    return true;
}

bool octorl::A2C::recvActorModel() {
    float *buffer = new float[actor.getElementCount()];    
    MPI_Recv(buffer, actor.getElementCount(), MPI_FLOAT, 0, octorl::model_tag, MPI_COMM_WORLD,
         MPI_STATUS_IGNORE);
    actor.loadFromSerial(buffer);
    //std::cout<<actor.parameters()[actor.parameters().size()-1]<<std::endl;
    delete[] buffer;
    return true;
}
void octorl::A2C::sendKeepRunning(bool run, int dst) {
    MPI_Send(&run, 1, MPI_C_BOOL, dst, octorl::keep_running_tag, MPI_COMM_WORLD);
}

void octorl::A2C::broadcastKeepRunning(bool run) {
    MPI_Bcast(&run, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
}

bool octorl::A2C::recvBroadcastKeepRunning() {
    bool run;
    MPI_Bcast(&run, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    return run;
}

bool octorl::A2C::recvKeepRunning() {
    bool kr;
    MPI_Recv(&kr, 1, MPI_C_BOOL, 0, octorl::keep_running_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    //std::cout<<"recieved "<<kr<<std::endl;
    return kr;
}

void octorl::A2C::addToLocalMemory(torch::Tensor init_obs, int act, float reward, float R, int done) {

    std::shared_ptr<float> buf(new float[env->getObservationSize() + 4]);
    int i = 0;
    
    init_obs = torch::flatten(init_obs);
    for(int j = 0; j < init_obs.numel(); j++)
        buf.get()[i++] = init_obs[j].item<float>();
    
    buf.get()[i++] = (float) act;
    buf.get()[i++] = reward;
    buf.get()[i++] = R;
    buf.get()[i++] = (float) done;

    local_memory.push_back(buf);
}


