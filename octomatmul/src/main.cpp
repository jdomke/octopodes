#include <iostream>
#include "denseMatrix.h"
#include "sparseMatrix.h"
#include "my_blas.h"
#include <time.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <quadmath.h>

using namespace std;

/*
args:
type - 0 for single precision GEMM/GEMV (BLAS), 1 for double precision GEMM/GEMV (BLAS), 2 for half precision GEMM (BLAS), 31 for dense matrix (2d vector implementation) multiplication, 32 for sparse matrix (map implementation) multiplication
batch_count - num of batches
batch_size - matrix count in each batch
m - row of matrix A and C
n - column of matrix B and C
k - column of matrix A and row of matrix B
layout - 0 for row major, 1 for column major
transA - 0 if A isn't transposed
transB - 0 if B isn't transposed
parallel - 1 for parallel
incx - increment for vector x
incy - increment for vector y
*/
typedef float myType;
const myType maxNum = 16;

int main(int argc, char* argv[]) {
  int type, batch_count, batch_size_, m_, n_, k_, layout_ = 0, transA_ = 0, transB_ = 0, parallel_ = 0, incx_ = 1, incy_ = 1;
  if (argc > 1) type = atoi(argv[1]);               //obtain information required to run tests
  if (argc > 2) batch_count = atoi(argv[2]);
  if (argc > 3) batch_size_ = atoi(argv[3]);
  if (argc > 4) m_ = atoi(argv[4]);
  if (argc > 5) n_ = atoi(argv[5]);
  if (argc > 6) k_ = atoi(argv[6]);
  if (argc > 7) layout_ = atoi(argv[7]);
  if (argc > 8) transA_ = atoi(argv[8]);
  if (argc > 9) transB_ = atoi(argv[9]); 
  if (argc > 10) parallel_ = atoi(argv[10]);
  if (argc > 11) incx_ = atoi(argv[11]);
  if (argc > 12) incy_ = atoi(argv[12]);
  
  assert(argc > 6);
  srand(10);

  cout << "type: " << type << "\nbatch_count: " << batch_count << "\nbatch_size: " << batch_size_ << "\nm: " << m_ << "\nn: " << n_ << "\nk: " << k_ << "\nlayout: "<< layout_ << "\ntransA: " << transA_ << "\ntransB: " << transB_ << "\nparallel: " << parallel_ <<  "\nincx_: " << incx_ << "\nincy_: " << incy_ << endl;
  if (type == 0 || type == 1 || type == 2){          //this entire code segment and its functions were inspired from https://github.com/wudu98/fugaku_batch_gemm

    CBLAS_LAYOUT layout = layout_ == 0 ? CblasRowMajor : CblasColMajor;       //set layout, transA, and transB according to commandline arguments
    CBLAS_TRANSPOSE transA = transA_ == 0 ? CblasNoTrans : CblasTrans;
    CBLAS_TRANSPOSE transB = transB_ == 0 ? CblasNoTrans : CblasTrans;

    size_t align = 256;
    int *batch_size    = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));      //allocate memory for variables,  variables are dynamic arrays of size batch_count so each batch has its own parameters for computation
    int *batch_head    = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));    //batch head indicate the start location of first matrix in each batch
    int *m      = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));
    int *n      = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));
    int *k      = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));
    int *lda    = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));
    int *ldb    = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));
    int *ldc    = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));
    int *incx   = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));
    int *incy   = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));
    int *lda_v = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));
    int total_batch_size = 0;
    
    #pragma omp parallel for schedule(static) reduction(+ : total_batch_size) if (parallel_ == 1)         //allocate memory on different threads
    for (int i = 0; i < batch_count; i++){
      batch_size[i] = batch_size_;
      total_batch_size += batch_size[i];        //increment total batch size based on the batch size of every batch
    }

    batch_head[0] = 0;
    for (int i = 1; i < batch_count; i++){          //batch head corresponds to the start location of the first matrix for each batch
      batch_head[i] = batch_size[i-1] + batch_head[i-1];    //calculate the batch head for each batch
    }

    if (type == 0){
      ProcessVariablesandPerformBLAS<float>(parallel_, m, n, k, transA, transB, layout, lda, ldb, ldc, lda_v, incx, incy, incx_, incy_, m_, k_, n_, batch_head, batch_count, batch_size, total_batch_size, maxNum);
    }
    else if (type == 1){
      ProcessVariablesandPerformBLAS<double>(parallel_, m, n, k, transA, transB, layout, lda, ldb, ldc, lda_v, incx, incy, incx_, incy_, m_, k_, n_, batch_head, batch_count, batch_size, total_batch_size, maxNum);
    }
    else if (type == 2){
      ProcessVariablesandPerformBLAS<MKL_F16>(parallel_, m, n, k, transA, transB, layout, lda, ldb, ldc, lda_v, incx, incy, incx_, incy_, m_, k_, n_, batch_head, batch_count, batch_size, total_batch_size, maxNum);
    }
    free(batch_size);
    free(batch_head);
    free(m);
    free(n);
    free(k);
    free(lda);
    free(ldb);
    free(ldc);
    free(incx);
    free(incy);
    free(lda_v);

  }
  else if (type == 31){
    // add code to handle parallel/no parallel, batch_count/batch_size, transA/transB
    DenseMatrix <myType> a(m_, k_);
    DenseMatrix <myType> b(k_, n_);
    for (int i = 0; i < m_; i++){
      for (int j = 0; j < k_; j++){
        a.insert(i, j, getRandomValue<myType>(0, maxNum));      //insert random elements to a and b 
      }
    }
    for (int i = 0; i < k_; i++){
      for (int j = 0; j < n_; j++){
        b.insert(i, j, getRandomValue<myType>(0, maxNum));
      } 
    }
    // cout << a << endl;
    // cout << b << endl;
    DenseMatrix <myType> c;
    double t0 = omp_get_wtime();
    c = a * b;
    double t1 = omp_get_wtime();
    cout << t1 - t0 << endl;

  }
  else if (type == 32){
    // add code to handle parallel/no parallel, batch_count/batch_size, transA/transB
    SparseMatrix <myType> a(m_, k_, rand() % ((m_*k_)/2), maxNum);
    SparseMatrix <myType> b(k_, n_, rand() % ((k_*n_)/2), maxNum);
    SparseMatrix <myType> c;
    // cout << a << endl;
    // cout << b << endl;
    double t0 = omp_get_wtime();
    c = a * b;
    double t1 = omp_get_wtime();
    cout << t1 - t0 << endl;
  }
  return 0;
}
