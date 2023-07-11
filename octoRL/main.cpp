//#include "include/driver.hpp"
//#include "include/quickTest.hpp"
#include "include/agents/DqnAsync.hpp"
#include "include/agents/A3C.hpp"
#include "include/agents/A2C.hpp"
//#include "include/Cnn.hpp"
#include "include/Policy.hpp"
//#include <libconfig.h++>
#include "include/envs/MountainCar.hpp"
//#include <c10d/ProcessGroupMPI.hpp>
#include <omp.h>
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include "mpi.h"

//#include "include/Mlp.hpp"


//using namespace libconfig;
using namespace std;
int main(int argc, char** argv) {
  int numranks, rank, comm_sz;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  int *anodes{new int[2]{64, 32}};
  shared_ptr<octorl::MountainCar> aenv(new octorl::MountainCar());
  vector<octorl::LayerInfo> obs_layer_info = {octorl::LayerInfo(octorl::linear, octorl::relu, "input",aenv->getObservationSize(),64),
    octorl::LayerInfo(octorl::linear, octorl::relu, "hl1",64,32),
    octorl::LayerInfo(octorl::linear, octorl::none, "output",32,1)
  };
  vector<octorl::LayerInfo> act_layer_info = {octorl::LayerInfo(octorl::linear, octorl::relu, "input",aenv->getObservationSize(),64),
    octorl::LayerInfo(octorl::linear, octorl::relu, "hl1",64,32),
    octorl::LayerInfo(octorl::linear, octorl::softmax, "output",32,aenv->getActionSize())
  };
  octorl::Policy pnet(obs_layer_info);
  octorl::Policy anet(act_layer_info);
  
  octorl::A2C async(aenv, 100000, pnet, anet, 0.99, 2000, 2314, 0.001, 32, rank, numranks);
  async.run();//action(aenv->reset().observation);
  MPI_Finalize();

  /*
  int rank, numranks, comm_sz;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);

  shared_ptr<octorl::MountainCar> aenv(new octorl::MountainCar());


  vector<octorl::LayerInfo> obs_layer_info = {octorl::LayerInfo(octorl::linear, octorl::relu, "input",aenv->getObservationSize(),64),
    octorl::LayerInfo(octorl::linear, octorl::relu, "hl1",64,32),
    octorl::LayerInfo(octorl::linear, octorl::none, "output",32,1)
  };
  vector<octorl::LayerInfo> act_layer_info = {octorl::LayerInfo(octorl::linear, octorl::relu, "input",aenv->getObservationSize(),64),
    octorl::LayerInfo(octorl::linear, octorl::relu, "hl1",64,32),
    octorl::LayerInfo(octorl::linear, octorl::softmax, "output",32,aenv->getActionSize())
  };
  //shared_ptr<octorl::Policy> anet = make_shared<octorl::Policy>(layer_info);
  //octorl::Mlp anet(aenv->getObservationSize(), aenv->getActionSize(), 2, anodes,1);
  //octorl::Mlp pnet(aenv->getObservationSize(), 1, 2, anodes);
  
  //shared_ptr<octorl::Policy> anet = make_shared<octorl::Policy>(act_layer_info);
  //shared_ptr<octorl::Policy> pnet = make_shared<octorl::Policy>(obs_layer_info);
  octorl::Policy pnet(obs_layer_info);
  octorl::Policy anet(act_layer_info);
  
  octorl::A2C async(aenv, 100000, pnet, anet, 0.99, 1500, 2314, 0.001, 16, rank, numranks);
  //octorl::A3C async(aenv, 100000, pnet, anet, 0.99, 1500, 2314, 0.001, 16, rank, numranks);

  //cout<<anet->parameters()<<endl;
  //return 0;
  //octorl::Policy anet(layer_info);
//  octorl::DqnAsync async(aenv,100000, anet, 0.95, 1,0.995,0.01, 500, 36456, 0.001,32, argc, argv);
  async.run();

//  // std::cout<<"ran: "<<rank<<std::endl;
 
//   async.test();
   MPI_Finalize();
  //std::cout<<anet.parameters()<<std::endl;
  //std::cout<<anet.conv2d_layers[0]<<std::endl;
  //std::cout<<anet.conv2d_layers[0]->forward(tensor)<<std::endl;
  //std::cout<<anet.forward(tensor)<<std::endl;
 
 // float *batch = new float[anet.getElementCount()];
 // anet.serialize(batch);
 // bnet.loadFromSerial(batch);


  // Config cfg;
  // cfg.lookup("name");cfg.readFile("../configs/test.cfg");
  // //string name = cfg.lookup("name");
  // const Setting& root = cfg.getRoot();
  // cout<<root["inventory"].exists("flarp")<<endl;
  // cout<<root["inventory"].exists("books")<<endl;



  vector<octorl::LayerInfo> layer_info1 = {octorl::LayerInfo(octorl::conv2d, octorl::relu,"in", 3,1,3),
    octorl::LayerInfo(octorl::conv2d, octorl::relu,"second", 32,32,3) ,
    octorl::LayerInfo(octorl::max_pool_2d, octorl::none,"pool",3,3,2),
    octorl::LayerInfo(octorl::flatten, octorl::none,"flat",3,1),
    octorl::LayerInfo(octorl::linear, octorl::softmax, "output",8192,3)};
  
  octorl::Policy anet(layer_info1);
  octorl::Policy bnet(layer_info1);
  torch::Tensor tensor = torch::rand({3,32,32});
  
  int *anodes{new int[2]{64, 32}};
  shared_ptr<octorl::MountainCar> aenv(new octorl::MountainCar());
  //shared_ptr<octorl::Cartpole> aenv(new octorl::Cartpole());
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  torch::Tensor tensor = torch::rand({aenv->getObservationSize()});

  async.run();//action(aenv->reset().observation);
  MPI_Finalize();

  return 0;
//  */
/*
  int rank, numranks, comm_sz;
  torch::Device device = torch::kCPU;
  if (torch::cuda::is_available()) {
    cout << "CUDA is available! Training on GPU." << endl;
    device = torch::kCUDA;
  }
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  int *anodes{new int[2]{64,32}};
  vector<std::shared_ptr<octorl::EnvironmentsBase>> actors;
  //cout<<*(argv+1)<<endl;
  for(int i = 0; i < numranks; i++) 
    actors.push_back( make_shared<octorl::Cartpole>(octorl::Cartpole(i))); 
  shared_ptr<octorl::Cartpole> aenv(new octorl::Cartpole(rank+2));
  octorl::Mlp anet(aenv->getObservationSize(), aenv->getActionSize(), 2, anodes);
  octorl::DqnAsync async(aenv,100000, anet, actors,0.95, 1,0.995,0.01, 500, 36456, 0.001,32, argc, argv);
  async.run();

 // std::cout<<"ran: "<<rank<<std::endl;
 
  async.test();
  MPI_Finalize();
  
  return 0;

  /*
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Status status;
  MPI_Request req1, req2; 

  cout<<numranks<<endl;

  cout<<"Rank: "<<rank<<endl;
   int *nodes{new int[2]{128,64}};
  shared_ptr<octorl::Cartpole> env(new octorl::Cartpole());
  shared_ptr<octorl::Cartpole> env2(new octorl::Cartpole());
  torch::Tensor tensor = torch::rand({env->getObservationSize()});
  octorl::Mlp net(env->getObservationSize(), env->getActionSize(), 1, nodes);
  octorl::Mlp net2(env->getObservationSize(), env->getActionSize(), 1, nodes);
//  octorl::loadstatedict(net2,net);

  octorl::Dqn agent(env,100000, net, 0.75, 1,0.995,0.01, 200, 36456, 0.001,5);
  octorl::Dqn agent2(env2,100000, net, 0.75, 1,0.995,0.01, 200, 36456, 0.001,5);



  return 0;
  */
}
