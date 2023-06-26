#include "../include/driver.hpp"


using namespace std;

void mountainCarDQNTest() {

    int *nodes{new int[2]{128,64}};
    shared_ptr<octorl::MountainCar> env(new octorl::MountainCar());
    torch::Tensor tensor = torch::rand({env->getObservationSize()});
    octorl::Mlp net(env->getObservationSize(),env->getActionSize(),1, nodes);
    octorl::Mlp net2(env->getObservationSize(),env->getActionSize(),1, nodes);
    octorl::Dqn agent(env,100000, net, 0.75, 1,0.995,0.01, 200, 36456, 0.001,5);
    torch::Tensor init_obs = env->reset().observation;
    int replay_start_size = 1000;
    auto act = agent.action(init_obs);
    auto obs = env->step(act.first);
    agent.pushToBuffer(octorl::Memory(env->currentStep(),init_obs,act.first,obs.reward, obs.observation, obs.done));
  
    int goals = 0;
    double reward_mean = 0, period_reward = 0;
    int i = 0;
    double losses =0;
    int loss_count = 0;

    for(i; i <100; i++) {
        double rewards = 0;
        while(!obs.done) {
            init_obs = obs.observation;
            act = agent.action(obs.observation);
            obs = env->step(act.first);
            rewards += obs.reward;
            agent.updateEpsilon();
            agent.pushToBuffer(octorl::Memory(env->currentStep(),init_obs,act.first,obs.reward, obs.observation, obs.done));
            if(agent.getBufferSize() >= agent.getBatchSize()){
                losses += agent.train();
                loss_count++;
            }
        }
        if(obs.goal)
            goals++;
        reward_mean += rewards;
        period_reward += rewards;
        obs = env->reset();
        if(i % 10 == 0) 
            agent.update_target();
        if(i % 10 == 0){
            cout<<"Ep: "<<i<<", Mean Reward: "<<reward_mean/(double)(i+1)<<", Period Reward: "<<period_reward/(double)(10)<<endl;
            cout<<"Loss Avg: "<<losses/loss_count<<endl;

            period_reward = 0;
        }

    }
    cout<<"Goals: "<<goals<<endl;
    cout<<"Ep: "<<i<<", Mean Reward: "<<reward_mean/(i+1)<<endl;
    cout<<"Test\n";
    int test_goals = 0;
    for(int i = 0; i < 100;i++) {
        obs = env->reset();
        while(!obs.done) {
            init_obs = obs.observation;
            auto action = agent.action(obs.observation);
            obs = env->step(action.first);
        }
        if(obs.goal)
            test_goals++;
    }
    cout<<"test Goals: "<<test_goals<<endl;
}

void blackjackDQNTest() {
    
    int *nodes{new int[2]{32,16}};
    shared_ptr<octorl::Blackjack> env(new octorl::Blackjack());
    
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32);
    //torch::Tensor test_tensor = torch::rand({{12,3,0}},options);
    
    
    octorl::Mlp net(env->getObservationSize(),env->getActionSize(),2, nodes);
    octorl::Mlp net2(env->getObservationSize(),env->getActionSize(),2, nodes);
    octorl::Dqn agent(env,1000, net, 0.8, 1,0.995,0.01, 10, 36456, 0.01,10);
    torch::Tensor init_obs = env->reset().observation;
    auto act = agent.action(init_obs);
    auto obs = env->step(act.first);
    agent.pushToBuffer(octorl::Memory(env->currentStep(),init_obs,act.first,obs.reward, obs.observation, obs.done));
    torch::Tensor test_tensor = init_obs;
    //torch::Tensor tensor = torch::rand({e->getObservationSize()});
    //cout<<"Test Tensor pre train: "<<test_tensor<<" "<<agent.modelPredict(test_tensor)<<endl;
    //return;
    int goals = 0;
    double reward_mean = 0, period_reward = 0;
    int i = 0;
    double losses =0;
    int loss_count = 0;
     for(i; i <10000; i++) {
        double rewards = 0;

        while(!obs.done) {
            init_obs = obs.observation;
            act = agent.action(obs.observation);
            obs = env->step(act.first);
           
            rewards += obs.reward;
            agent.updateEpsilon();
            agent.pushToBuffer(octorl::Memory(env->currentStep(),init_obs,act.first,obs.reward, obs.observation, obs.done));
            if(agent.getBufferSize() >= agent.getBatchSize()){
                
                
                //agent.printTargetModelParameters();
                losses += agent.train();
                loss_count++;
                //agent.modelsMatch();
            
            }
        }
        //agent.updateEpsilon();
        if(rewards >= 1)
            goals++;
        //cout<<rewards<<endl;
        reward_mean += rewards;
        period_reward += rewards;
        //cout<<"\n done \n";
        obs = env->reset();
        //agent.updateEpsilon();
        if(i % 32 == 0) {
            //cout<<"Test Tensor pre train: "<<test_tensor<<" "<<agent.modelPredict(test_tensor)<<endl;
            agent.update_target();
        }
        if(i % 1000 == 0){
            cout<<"Ep: "<<i<<", Mean Reward: "<<reward_mean/(double)(i+1)<<", Period Reward: "<<period_reward/(double)(1000)<<endl;
            period_reward = 0;
            cout<<"Loss Avg: "<<losses/loss_count<<endl;
        }

    }
    cout<<"Wins: "<<goals<<endl;

    int test_goals = 0;
    for(int i = 0; i < 100; i++){
        int rewards = 0; 
        obs = env->reset();
        while(!obs.done) {
            init_obs = obs.observation;
            auto action = agent.action(obs.observation);
            obs = env->step(action.first);
            rewards += obs.reward;
        }
        if(rewards >= 1)
            test_goals++;
    }

    cout<<"test Goals: "<<test_goals<<endl;

}

void cartpoleDQNTest() {
    int net_size = 1;
    int *nodes{new int[net_size]{64}};
    shared_ptr<octorl::Cartpole> env(new octorl::Cartpole());
    torch::Tensor tensor = torch::rand({env->getObservationSize()});
    octorl::Mlp net(env->getObservationSize(),env->getActionSize(),net_size, nodes);
    octorl::Mlp net2(env->getObservationSize(),env->getActionSize(),net_size, nodes);
    octorl::Dqn agent(env,10000, net, 0.95, 1,0.995,0.01, 500, 36456, 0.0001,32);
    torch::Tensor init_obs = env->reset().observation;
    int replay_start_size = 1000;
    auto act = agent.action(init_obs);
    auto obs = env->step(act.first);
    agent.pushToBuffer(octorl::Memory(env->currentStep(),init_obs,act.first,obs.reward, obs.observation, obs.done));
  
    int goals = 0;
    double reward_mean = 0, period_reward = 0;
    int length = 0;
    int i = 0;
    double losses =0;
    int loss_count = 0;

    for(i; i <10000; i++) {
        double rewards = 0;
        while(!obs.done) {
            init_obs = obs.observation;
            act = agent.action(obs.observation);
            obs = env->step(act.first);
            rewards += obs.reward;
            agent.pushToBuffer(octorl::Memory(env->currentStep(),init_obs,act.first,obs.reward, obs.observation, obs.done));
            if(agent.getBufferSize() >= agent.getBatchSize()){
                agent.updateEpsilon();
                losses += agent.train();
                loss_count++;
            }
        }
        if(obs.goal)
            goals++;
        reward_mean += rewards;
        period_reward += rewards;
        length += env->currentStep();
        obs = env->reset();
        if(i % 3 == 0) 
            agent.update_target();
        if(i % 100 == 0){
            cout<<"Ep: "<<i<<", Mean Reward: "<<reward_mean/(double)(i+1)<<", Period Reward: "<<period_reward/(double)(100)<<", Avg steps "<<length/(double)100<<endl;
            cout<<"Loss Avg: "<<losses/loss_count<<endl;
            length = 0;
            period_reward = 0;
        }

    }
    cout<<"Goals: "<<goals<<endl;
    cout<<"Ep: "<<i<<", Mean Reward: "<<reward_mean/(i+1)<<endl;
    cout<<"Test\n";
    int test_goals = 0;
    length = 0;
    for(int i = 0; i < 100;i++) {
        obs = env->reset();
        while(!obs.done) {
            init_obs = obs.observation;
            auto action = agent.action(obs.observation);
            obs = env->step(action.first);
        }
        if(obs.goal)
            test_goals++;
        
        length += env->currentStep();
    
    }
    cout<<"test Goals: "<<test_goals<<endl;
    cout<<"Avg run time: "<<length/(double)100<<endl;
}

void openMpFirstTry() {

     int i, tid;
    int *nodes{new int[2]{128,64}};
    shared_ptr<octorl::MountainCar> env(new octorl::MountainCar());
    shared_ptr<octorl::MountainCar> env1(new octorl::MountainCar());
    shared_ptr<octorl::MountainCar> env2(new octorl::MountainCar());
    shared_ptr<octorl::MountainCar> env3(new octorl::MountainCar());
    torch::Tensor tensor = torch::rand({env->getObservationSize()});
    octorl::Mlp net(env->getObservationSize(),env->getActionSize(),1, nodes);
    octorl::Mlp net2(env->getObservationSize(),env->getActionSize(),1, nodes);
    octorl::Dqn agent(env,100000, net, 0.75, 1,0.995,0.01, 200, 36456, 0.001,32);

    pair<int,int> act;
    octorl::StepReturn obs;
    torch::Tensor init_obs;
    omp_lock_t writelock;
    double total_rewards = 0;

    omp_init_lock(&writelock);
    int ep = 0;
    int steps = 0;
    int goals = 0;
    while(ep < 500) {
      #pragma omp barrier 
      {
        #pragma omp parallel num_threads(2) shared(agent, env, env1, env2, env3, total_rewards,steps, goals) private(i, tid,act, obs, init_obs)
        {
        
          omp_set_nested(2);
          tid=omp_get_thread_num();
            #pragma omp sections 
            {
                
                #pragma omp section 
                {
                  int r = 0;
                  init_obs = env->reset().observation;  
                  act = agent.action(init_obs);
                  obs = env->step(act.first);
                  omp_set_lock(&writelock);
                  
                  steps++;
                  total_rewards += obs.reward;
                  agent.pushToBuffer(octorl::Memory(env->currentStep(),init_obs,act.first,obs.reward, obs.observation, obs.done));
                  omp_unset_lock(&writelock);
                  while(!obs.done) {
                    init_obs = obs.observation;
                    act = agent.action(obs.observation);
                    obs = env->step(act.first);
                    agent.updateEpsilon();

                    omp_set_lock(&writelock);
                    total_rewards += obs.reward;
                    steps++;
                    agent.pushToBuffer(octorl::Memory(env->currentStep(),init_obs,act.first,obs.reward, obs.observation, obs.done));
                    omp_unset_lock(&writelock);
                  }
                  if(obs.goal) {
                    omp_set_lock(&writelock);
                    goals++;
                    omp_unset_lock(&writelock);
                  }
                }
                
                #pragma omp section
                {
                  init_obs = env1->reset().observation;  
                  act = agent.action(init_obs);
                  obs = env1->step(act.first);
                  omp_set_lock(&writelock);
                  total_rewards += obs.reward;
                  steps++;
                  agent.pushToBuffer(octorl::Memory(env1->currentStep(),init_obs,act.first,obs.reward, obs.observation, obs.done));
                      omp_unset_lock(&writelock);
                  while(!obs.done) {
                    init_obs = obs.observation;
                    act = agent.action(obs.observation);
                    obs = env1->step(act.first);
                    agent.updateEpsilon();
                    omp_set_lock(&writelock);
                    total_rewards += obs.reward;
                    steps++;
                    agent.pushToBuffer(octorl::Memory(env1->currentStep(),init_obs,act.first, obs.reward, obs.observation, obs.done));
                    omp_unset_lock(&writelock);
                  }
                  if(obs.goal) {
                    omp_set_lock(&writelock);
                    goals++;
                    omp_unset_lock(&writelock);
                  }
                    
                }

            }
        }
      }
      ep++;
      if(ep %6 == 0) {
        cout<<"mean rewards: "<<total_rewards/(6*2)<<endl;
        cout<<"Goals: "<<goals<<endl;
        goals = 0;
        total_rewards = 0;
        for(int i = 0; i < 10; i++) {
              agent.train();
        }
        agent.update_target();
      }
    }
   cout<<"ASDFSA\n";
    cout<<agent.getBufferSize()<<endl;
    
    return;
}
