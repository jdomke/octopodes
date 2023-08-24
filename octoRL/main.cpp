#include "include/agents/DqnAsync.hpp"
#include "include/agents/A3C.hpp"
#include "include/agents/A2C.hpp"
#include "include/Policy.hpp"
#include <libconfig.h++>
#include "include/envs/MountainCar.hpp"
#include "include/envs/CNNTest.hpp"
#include <omp.h>
#include <time.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <map>
#include "mpi.h"
#include "include/configure.hpp"
#include <map>
#include <papi.h>
#include <cstring>
#include <ctime>

using namespace libconfig;
using namespace std;

void handle_error (int retval)
 {
      printf("PAPI error %d: %s\n", retval, PAPI_strerror(retval));
      exit(1);
}


int main(int argc, char** argv) {


  int numranks, rank, comm_sz;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int retval, EventSet = PAPI_NULL;
	 int Events[3] = {PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_BR_INS};
	 long_long values[3];

 retval = PAPI_library_init(PAPI_VER_CURRENT);
 if(rank == 0)
	 cout<<"StartTime: "<<time(NULL)<<endl;
 if (retval != PAPI_VER_CURRENT) {
   fprintf(stderr, "PAPI library init error!\n");
   exit(1);
 }

 if (PAPI_create_eventset(&EventSet) != PAPI_OK)
     handle_error(1);


 if (PAPI_add_events(EventSet, Events, 3) != PAPI_OK)
     handle_error(1);


 if (PAPI_start(EventSet) != PAPI_OK)
     handle_error(1);
 

  if(argc >= 2)
    configureAndRun(argv[1]);

	if (PAPI_stop(EventSet, values) != PAPI_OK)
            handle_error(1);
  // cout<<"\nValues for rank "<<rank<<"\n\tCyc: "<<values[0]<<"\n\tIns: "<<values[1]<<"\n\tBr: "<<values[2]<<
//	   "\n\tIPC: "<<real(values[1])/real(values[0])<<endl;
   	
	cout<<"Values for rank "<<rank<<" Cyc: "<<values[0]<<" Ins: "<<values[1]<<" Br: "<<values[2]<<
	   " IPC: "<<real(values[1])/real(values[0])<<endl;
	MPI_Finalize();
 return 0;
}
