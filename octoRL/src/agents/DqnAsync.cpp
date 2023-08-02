#include "../../include/agents/DqnAsync.hpp"


octorl::DqnAsync::DqnAsync(std::shared_ptr<octorl::EnvironmentsBase> environment, size_t buffer_size, octorl::Policy policy_model,
                float g, float eps, float decay, float eps_min,int ep_count, int seed, double lr,int batch,int freq, int r, int nr) {

    env = environment;
    exp_buffer = ExperienceReplay(buffer_size);
    local_size = buffer_size;
    gamma = g;
    epsilon = eps;
    epsilon_decay = decay;
    epsilon_min = eps_min;
    episodes = ep_count;
    model = policy_model;
    batch_freq = freq;
    epochs = ep_count;
    target_model = policy_model;
    
    if (torch::cuda::is_available() && r == 0) { 
	std::cout << "CUDA is available! Training on GPU." << r<<std::endl;                                                                                                                                           
        device = torch::kCUDA;                                                                                                                                                                                     
    }         
    model.to(device); 
    target_model.to(device); 
    
    octorl::loadstatedict2(target_model,policy_model);
    octorl::loadstatedict2(model,policy_model);
    learning_rate = lr;
    batch_size = batch;
    local_batch = batch;
    model_optimizer = std::make_shared<torch::optim::Adam>(model.parameters(), lr);
    target_optimizer = std::make_shared<torch::optim::Adam>(target_model.parameters(), lr);
    srand(seed);
    rank = r;
    numranks = nr;
    losses = torch::zeros(100*(numranks-1));
    loss_counter = 0;
}

void octorl::DqnAsync::test() {
    std::cout<<"test on rank: "<<rank<<std::endl;
    epsilon = 0;
        int test_goals = 0;
        float avg_reward= 0;
        torch::Tensor init_obs;
        for(int i = 0; i < 100; i++){
            int rewards = 0; 
            auto obs = env->reset();    
            rewards += obs.reward;
            avg_reward += obs.reward;
            int act;
            while(!obs.done) {
		obs.observation.to(device);
                //init_obs = obs.observation;
		act = action(obs.observation.to(device),true);
                obs = env->step(act);
                rewards += obs.reward;
                avg_reward += obs.reward;
            }
            if(obs.goal)
                test_goals++;
        }

    std::cout<<"test Goals: "<<test_goals<<std::endl;
    std::cout<<"Avg Reward: "<<avg_reward/100<<std::endl;
}

void octorl::DqnAsync::learnerRun() {
    MPI_Barrier(MPI_COMM_WORLD);
    broadcastModel();
    broadcastKeepRunning(true);
    for(int g = 0; g < epochs; g++) {
        std::vector<int> to_send;
        
        for(int i = 1; i < numranks; i++){
            recvBatchAndTrain();
        }
        updateTarget();
        MPI_Barrier(MPI_COMM_WORLD);
        broadcastModel();
        broadcastKeepRunning(true);
        if(g % 100 == 0) {
            std::cout<<"Epoch: "<<g<<std::endl;
      	    test();
	    std::cout<<"Average Loss: "<<torch::mean(losses)<<std::endl;
            loss_counter = 0;
	}
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    broadcastModel();
    broadcastKeepRunning(false);
}

void octorl::DqnAsync::workerRun() {
    MPI_Barrier(MPI_COMM_WORLD);
    recvBroadcastModel();
    std::cout<<"worker runn\n";
    int g = 0;
    while(recvBroadcastKeepRunning()){
        torch::Tensor init_obs = env->reset().observation.to(device);
        
        auto act = action(init_obs);
        auto obs = env->step(act);
	obs.observation.to(device);
        addToLocalMemory(init_obs, act, obs.reward, obs.observation.to(device), obs.done);
        int steps = 1;
        int running_reward = obs.reward;
        while(!obs.done) {
	    obs.observation.to(device);
            init_obs = obs.observation.to(device);
            act = action(init_obs);
            obs = env->step(act);
            addToLocalMemory(init_obs, act, obs.reward, obs.observation.to(device), obs.done);
            running_reward += obs.reward;
            steps++;
        }
        epsilon *= epsilon_decay;
        epsilon = std::max(epsilon, epsilon_min);
 	sendBatch(obs.done);
        MPI_Barrier(MPI_COMM_WORLD);
        recvBroadcastModel();
    }
}

void octorl::DqnAsync::run() {
    if(rank == 0) {
        learnerRun();
    }
    else {
        workerRun();
    }
}

void octorl::DqnAsync::sendKeepRunning(bool run, int dst) {
    MPI_Send(&run, 1, MPI_C_BOOL, dst, octorl::keep_running_tag, MPI_COMM_WORLD);
}

void octorl::DqnAsync::broadcastKeepRunning(bool run) {
    MPI_Bcast(&run, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
}

bool octorl::DqnAsync::recvKeepRunning() {
    bool kr;
    MPI_Recv(&kr, 1, MPI_C_BOOL, 0, octorl::keep_running_tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return kr;
}

bool octorl::DqnAsync::recvBroadcastKeepRunning() {
    bool kr;
    MPI_Bcast(&kr, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    return kr;
}

int octorl::DqnAsync::action(torch::Tensor state, bool testing) {

    float r = distribution(gen);

    if(r < epsilon && !testing) {
        int r = rand() % env->getActionSize();
        return r;
    }
    return model.forward(state).argmax().item().toInt();
}

bool octorl::DqnAsync::sendModel(int r) {
    float *buffer = new float[model.getElementCount()];    
    model.serialize(buffer);
    
    MPI_Send(buffer, model.getElementCount(), MPI_FLOAT, r, octorl::model_tag, MPI_COMM_WORLD);
    delete[] buffer;
    return true;
}

bool octorl::DqnAsync::broadcastModel() {
    float *buffer = new float[model.getElementCount()];    
    model.serialize(buffer);
    
    MPI_Bcast(buffer, model.getElementCount(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    delete[] buffer;
    return true;
}

bool octorl::DqnAsync::recvBroadcastModel() {
    float *buffer = new float[model.getElementCount()];    
    
    MPI_Bcast(buffer, model.getElementCount(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    model.loadFromSerial(buffer);
    delete[] buffer;
    return true;
}

bool octorl::DqnAsync::recvModel() {

    float *buffer = new float[model.getElementCount()];    
    MPI_Recv(buffer, model.getElementCount(), MPI_FLOAT, 0, octorl::model_tag, MPI_COMM_WORLD,
         MPI_STATUS_IGNORE);
    model.loadFromSerial(buffer);
    delete[] buffer;
    return true;
}

bool octorl::DqnAsync::sendBatch(int done) {
    float *batch = new float[local_batch * env->memorySize() + 2];
    int b = 0;
    batch[b++] = rank;
    batch[b++] = done;
    std::deque<std::shared_ptr<float>> sample_set;
    auto gen1 = std::mt19937 {std::random_device {}()};
    std::sample(local_memory.begin(), local_memory.end(),std::back_inserter(sample_set), local_batch, gen1);
     
    for(int i = 0; i < sample_set.size(); i++) {
        for(int j = 0; j < env->memorySize(); j++){
            batch[b++] = sample_set[i].get()[j];
        }
    }
    MPI_Send(batch, local_batch * env->memorySize()+2, MPI_FLOAT, 0, octorl::batch_tag, MPI_COMM_WORLD);

    delete[] batch;

    return true;    
}

// unused
std::pair<int,int> octorl::DqnAsync::recvBatch() {
    float *batch = new float[local_batch * env->memorySize() + 2];
 
    MPI_Recv(batch,local_batch * env->memorySize() + 2, MPI_FLOAT, MPI_ANY_SOURCE, octorl::batch_tag, MPI_COMM_WORLD,
    MPI_STATUS_IGNORE);

    torch::Tensor init_obs = torch::ones({{env->getObservationSize()}});
    torch::Tensor next_obs = torch::ones({{env->getObservationSize()}});
    int act, done;
    float reward;
    int b = 0;
    int src = (int) batch[b++];
    int finished = (int) batch[b++];
    for(int i = 0; i < local_batch; i++) {
        for(int j = 0; j < env->getObservationSize(); j++){
            init_obs[j] = batch[b++];
        }
        
        init_obs = env->shapeObservation(init_obs);

        act = (int) batch[b++];
        reward = batch[b++];
        for(int j = 0; j < env->getObservationSize(); j++){
            next_obs[j] = batch[b++];
        }
        next_obs = env->shapeObservation(next_obs);
        done = (int) batch[b++];

        pushToBuffer(octorl::Memory(-1,init_obs, act, reward, next_obs,(bool) done));
    }

    delete[] batch;
    return std::make_pair(src,finished);
}



int octorl::DqnAsync::recvBatchAndTrain() {

    float *batch = new float[local_batch * env->memorySize() + 2];
    std::vector<octorl::Memory> buf;
    MPI_Recv(batch,local_batch * env->memorySize() + 2, MPI_FLOAT, MPI_ANY_SOURCE, octorl::batch_tag, MPI_COMM_WORLD,
    MPI_STATUS_IGNORE);
    
    // decode mpi buffer
    int act, done;
    float reward;
    int b = 0;
    int src = (int) batch[b++];
    int finished = (int) batch[b++];
    for(int i = 0; i < local_batch; i++) {
        torch::Tensor init_obs = torch::ones((env->getObservationSize()));
        torch::Tensor next_obs = torch::ones((env->getObservationSize()));
        for(int j = 0; j < env->getObservationSize(); j++){
            init_obs[j] = batch[b++];
        }
        init_obs = env->shapeObservation(init_obs).to(device);
        act = (int) batch[b++];
        reward = batch[b++];
        for(int j = 0; j < env->getObservationSize(); j++){
            next_obs[j] = batch[b++];
        }
        next_obs = env->shapeObservation(next_obs).to(device);
        done = (int) batch[b++];

        buf.push_back(octorl::Memory(-1,init_obs, act, reward, next_obs,(bool) done));
    }
    losses[loss_counter++] = trainOnBatch(buf);
    delete[] batch;
    return src;
}

float octorl::DqnAsync::trainOnBatch(std::vector<octorl::Memory> batch) {
    std::vector<torch::Tensor> in_vec;
    std::vector<torch::Tensor> out_vec;
//    std::cout<<"train\n";
    for(auto i : batch){
	i.state.to(device);
        in_vec.push_back(i.state);
        out_vec.push_back(calcTargetF(i));
    }

    torch::TensorList input {in_vec};
    torch::TensorList target {out_vec};
    torch::Tensor input_batch = torch::cat(input).to(device);
    torch::Tensor target_batch = torch::cat(target).to(device);
    model_optimizer->zero_grad();
    torch::Tensor output = model.forward({input_batch});

    torch::Tensor loss = torch::mse_loss(target_batch,output).to(device);
    loss.backward({}, false);
    model_optimizer->step();
    return loss.item().toDouble();
}

torch::Tensor octorl::DqnAsync::calcTargetF(octorl::Memory m) {
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

// encodes memory into array of floats to easily send over mpi
void octorl::DqnAsync::addToLocalMemory(torch::Tensor init_obs, int act, float reward, torch::Tensor next_obs, int done) {
    std::shared_ptr<float> buf(new float[env->memorySize()]);
    int i = 0;
    init_obs = torch::flatten(init_obs);
    for(int j = 0; j < init_obs.numel(); j++)
        buf.get()[i++] = init_obs[j].item<float>();
    
    buf.get()[i++] = (float) act;
    buf.get()[i++] = reward;
    next_obs = torch::flatten(next_obs);
    for(int j = 0; j < next_obs.numel(); j++)
        buf.get()[i++] = next_obs[j].item<float>();
    buf.get()[i++] = (float) done;

    if(local_memory.size() == local_size) {
        local_memory.pop_front();
        local_memory.push_back(buf);
    }
    else
        local_memory.push_back(buf);
}

void octorl::DqnAsync::pushToBuffer(octorl::Memory m) {
    exp_buffer.addToReplayBuffer(m);
}

void octorl::DqnAsync::updateTarget() {
    octorl::loadstatedict2(target_model,model);
}
