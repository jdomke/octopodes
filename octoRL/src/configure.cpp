#include "../include/configure.hpp"

using namespace libconfig;
using namespace std;


map<string, octorl::layer_type> layer_map {
        {"linear",octorl::linear},
        {"conv2d", octorl::conv2d},
        {"max_pool_2d", octorl::max_pool_2d},
        {"flatten", octorl::flatten}};

map<string, octorl::activation_type> activation_map {
        {"relu",octorl::relu},
        {"softmax", octorl::softmax},
        {"none", octorl::none}};

void configureAndRun(const char* file) {
    Config cfg;
    cfg.readFile(file);

    const Setting& root = cfg.getRoot();
    
    if(root.exists("a2c")) {
        a2cConfigureAndRun(root); }
    else if(root.exists("a3c")) {
        a3cConfigureAndRun(root); }
    else if(root.exists("dqnAsync")) { 
        dqnAsyncConfigureAndRun(root); }
    
}

void a2cConfigureAndRun(const Setting& root) {
    
    const Setting& a2c_settings = root["a2c"];
    const Setting& actor = a2c_settings["actor"];
    const Setting& critic = a2c_settings["critic"];

    octorl::Policy actor_net = modelParse(actor);
    octorl::Policy critic_net = modelParse(critic);

    int rank, numranks, buffer_size = 100000, episode_count = 500, seed = 29384, batch = 32;
    float learning_rate = 0.001, gamma = 0.99;
    string env =(const char *) root.lookup("environment");
    transform(env.begin(), env.end(), env.begin(), ::toupper);
    a2c_settings.lookupValue("buffer_size", buffer_size);
    a2c_settings.lookupValue("episode_count", episode_count);
    a2c_settings.lookupValue("seed", seed);
    a2c_settings.lookupValue("batch_size", batch);
    a2c_settings.lookupValue("learning_rate", learning_rate);
    a2c_settings.lookupValue("gamma", gamma);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(env == "MOUNTAINCAR") {
        octorl::A2C async(make_shared<octorl::MountainCar>(), buffer_size, critic_net, actor_net, gamma, episode_count,
            seed, learning_rate, batch, rank, numranks); 
        async.run();
    }
    else if(env == "CARTPOLE") {
        octorl::A2C async(make_shared<octorl::Cartpole>(), buffer_size, critic_net, actor_net, gamma, episode_count,
            seed, learning_rate, batch, rank, numranks);
        async.run();
    }
    
}

void a3cConfigureAndRun(const libconfig::Setting& root) {
   
    const Setting& a3c_settings = root["a3c"];
    const Setting& actor = a3c_settings["actor"];
    const Setting& critic = a3c_settings["critic"];

    octorl::Policy actor_net = modelParse(actor);
    octorl::Policy critic_net = modelParse(critic);

    int rank, numranks, buffer_size = 100000, episode_count = 500, seed = 29384, batch = 32;
    float learning_rate = 0.001, gamma = 0.99;
    string env =(const char *) root.lookup("environment");
    transform(env.begin(), env.end(), env.begin(), ::toupper);
    a3c_settings.lookupValue("buffer_size", buffer_size);
    a3c_settings.lookupValue("episode_count", episode_count);
    a3c_settings.lookupValue("seed", seed);
    a3c_settings.lookupValue("batch_size", batch);
    a3c_settings.lookupValue("learning_rate", learning_rate);
    a3c_settings.lookupValue("gamma", gamma);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(env == "MOUNTAINCAR") {
        octorl::A3C async(make_shared<octorl::MountainCar>(), buffer_size, critic_net, actor_net, gamma, episode_count,
            seed, learning_rate, batch, rank, numranks); 
        async.run();
    }
    else if(env == "CARTPOLE") {
        octorl::A3C async(make_shared<octorl::Cartpole>(), buffer_size, critic_net, actor_net, gamma, episode_count,
            seed, learning_rate, batch, rank, numranks);
        async.run();
    }
    

}

void dqnAsyncConfigureAndRun(const libconfig::Setting& root) {
 
    const Setting& dqn_settings = root["dqnAsync"];
    const Setting& policy = dqn_settings["policy"];
    
    octorl::Policy policy_net = modelParse(policy);

    int rank, numranks, buffer_size = 100000, episode_count = 500, seed = 29384, batch = 32, batch_freq = 32;
    float learning_rate = 0.001, gamma = 0.99, epsilon = 1, epsilon_min = 0.01, decay = 0.95;
    string env =(const char *) root.lookup("environment");
    transform(env.begin(), env.end(), env.begin(), ::toupper);

    dqn_settings.lookupValue("buffer_size", buffer_size);
    dqn_settings.lookupValue("episode_count", episode_count);
    dqn_settings.lookupValue("seed", seed);
    dqn_settings.lookupValue("batch_size", batch);
    dqn_settings.lookupValue("batch_freq", batch_freq);
    dqn_settings.lookupValue("learning_rate", learning_rate);
    dqn_settings.lookupValue("gamma", gamma);
    dqn_settings.lookupValue("epsilon", epsilon);
    dqn_settings.lookupValue("epsilon_min", epsilon_min);
    dqn_settings.lookupValue("epsilon_decay", decay);
    MPI_Comm_size(MPI_COMM_WORLD, &numranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if(env == "MOUNTAINCAR") {
        octorl::DqnAsync async(make_shared<octorl::MountainCar>(), buffer_size, policy_net, gamma, epsilon, decay, epsilon_min, 
            episode_count, seed, learning_rate, batch, batch_freq,rank, numranks); 
        async.run();
    }
    else if(env == "CARTPOLE") {
        octorl::DqnAsync async(make_shared<octorl::Cartpole>(), buffer_size, policy_net, gamma, epsilon, decay, epsilon_min, 
            episode_count, seed, learning_rate, batch, batch_freq,rank, numranks); 
        async.run();
    }

}

octorl::LayerInfo layerParse(const Setting& layer) {
    octorl::layer_type type;
    octorl::activation_type activation;
    string  label;
    int input, output, kernel_size = 1, padding = 1, dilation = 1;

    type = layer_map[(const char *)layer.lookup("type")];
    activation = activation_map[(const char *)layer.lookup("activation")];
    label = (const char *)layer.lookup("label");
    layer.lookupValue("input", input);
    layer.lookupValue("output", output);

    //if(layer.exists("kernel_size"))
    layer.lookupValue("kernel_size", kernel_size);
    
    //if(layer.exists("padding"))
    layer.lookupValue("padding", kernel_size);
    
    //if(layer.exists("dilation"))
    layer.lookupValue("dilation", kernel_size);
    
    return octorl::LayerInfo(type, activation, label, input, output, kernel_size, padding, dilation);
}

octorl::Policy modelParse(const Setting& model){
    vector<octorl::LayerInfo> layer_info;
    for(int i = 0; i < model.getLength(); i++) 
        layer_info.push_back(layerParse(model[i]));

    return octorl::Policy(layer_info);
}

