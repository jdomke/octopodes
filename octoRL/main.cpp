#include "include/driver.hpp"
#include "include/quickTest.hpp"
#include "include/agents/DqnAsync.hpp"
#include "include/agents/A3C.hpp"
//#include <c10d/ProcessGroupMPI.hpp>
#include <omp.h>
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include "mpi.h"



using namespace std;
int main(int argc, char** argv) {//*/
  int *anodes{new int[2]{24,24}};
  int rank, numranks, comm_sz;
  //shared_ptr<octorl::MountainCar> aenv(new octorl::MountainCar());
  shared_ptr<octorl::Cartpole> aenv(new octorl::Cartpole());
    MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  octorl::Mlp anet(aenv->getObservationSize(), aenv->getActionSize(), 2, anodes,1);
  octorl::Mlp pnet(aenv->getObservationSize(), 1, 2, anodes);

  octorl::A3C async(aenv, 100000, pnet, anet, 0.95, 500, 2314, 0.001, 64, rank, numranks);
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