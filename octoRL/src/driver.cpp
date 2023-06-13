#include "../include/driver.hpp"


using namespace std;

void mountainCarDQNTest() {

    int *nodes{new int[2]{64,32}};
    shared_ptr<octorl::MountainCar> env(new octorl::MountainCar());
    torch::Tensor tensor = torch::rand({env->getObservationSize()});
    octorl::Mlp net(env->getObservationSize(),env->getActionSize(),1, nodes);
    octorl::Mlp net2(env->getObservationSize(),env->getActionSize(),1, nodes);
    //  for(auto x : net.parameters()){
    //     cout<<x<<endl;
    // }
    //octorl::Mlp net2(env->getObservationSize(),env->getActionSize(),0, nodes);// = net;
    octorl::Dqn agent(env,100000, net, net2, 0.95, 1,0.995,0.01, 200, 36456, 0.001,100);
    torch::Tensor init_obs = env->reset().observation;
    int replay_start_size = 1000;
    auto act = agent.action(init_obs);
    auto obs = env->step(act.first);
    agent.pushToBuffer(octorl::Memory(env->currentStep(),init_obs,act,obs.reward, obs.observation, obs.done));
 
    //agent.printModelParameters();
  
    int goals = 0;
    double reward_mean = 0, period_reward = 0;
    int i = 0;
    double losses =0;
    int loss_count = 0;

    for(i; i <100; i++) {
        double rewards = 0;
        while(!obs.done) {
            init_obs = obs.observation;
            auto action = agent.action(obs.observation);
            obs = env->step(action.first);
           // if(i%100 == 0)
             //   cout<<"Position at step "<<env->currentStep()<<": "<<obs.observation<<endl;
            rewards += obs.reward;
            agent.updateEpsilon();
            agent.pushToBuffer(octorl::Memory(env->currentStep(),init_obs,act,obs.reward, obs.observation, obs.done));
            if(agent.getBufferSize() >= agent.getBatchSize()){
                losses += agent.train();
                loss_count++;
            }
        }
        if(obs.goal)
            goals++;
        //cout<<rewards<<endl;
        reward_mean += rewards;
        period_reward += rewards;
        //cout<<"\n done \n";
        obs = env->reset();
        //agent.updateEpsilon();
        if(i % 10 == 0) 
            agent.update_target();
        //}
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
    octorl::Dqn agent(env,1000, net, net2, 0.8, 1,0.995,0.01, 10, 36456, 0.01,10);
    torch::Tensor init_obs = env->reset().observation;
    auto act = agent.action(init_obs);
    auto obs = env->step(act.first);
    agent.pushToBuffer(octorl::Memory(env->currentStep(),init_obs,act,obs.reward, obs.observation, obs.done));
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
            auto action = agent.action(obs.observation);
            obs = env->step(action.first);
           
            rewards += obs.reward;
            agent.updateEpsilon();
            agent.pushToBuffer(octorl::Memory(env->currentStep(),init_obs,act,obs.reward, obs.observation, obs.done));
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
}