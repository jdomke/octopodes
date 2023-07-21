//#include "include/driver.hpp"
//#include "include/quickTest.hpp"
#include "include/agents/DqnAsync.hpp"
#include "include/agents/A3C.hpp"
#include "include/agents/A2C.hpp"
//#include "include/Cnn.hpp"
#include "include/Policy.hpp"
#include <libconfig.h++>
#include "include/envs/MountainCar.hpp"
#include "include/envs/CNNTest.hpp"
//#include <c10d/ProcessGroupMPI.hpp>
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
//#include "include/Mlp.hpp"


using namespace libconfig;
using namespace std;

// void handle_error (int retval)
// {
//      printf("PAPI error %d: %s\n", retval, PAPI_strerror(retval));
//      exit(1);
// }


int main(int argc, char** argv) {


  int numranks, rank, comm_sz;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // int retval, EventSet = PAPI_NULL;
	// int Events[3] = {PAPI_TOT_CYC, PAPI_TOT_INS, PAPI_BR_INS};
	// long_long values[3];

	// retval = PAPI_library_init(PAPI_VER_CURRENT);

	// if (retval != PAPI_VER_CURRENT) {
	//   fprintf(stderr, "PAPI library init error!\n");
	//   exit(1);
	// }

	// if (PAPI_create_eventset(&EventSet) != PAPI_OK)
	//     handle_error(1);


	// if (PAPI_add_events(EventSet, Events, 3) != PAPI_OK)
	//     handle_error(1);


	// if (PAPI_start(EventSet) != PAPI_OK)
	//     handle_error(1);
 

  if(argc >= 2)
    configureAndRun(argv[1]);
  
  MPI_Finalize();
 return 0;
}
