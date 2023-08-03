#include <iostream>
#include <map>
#include <random>
#include <time.h>
#include <utility>
#include <omp.h>
#include <algorithm>
#include <type_traits>
#include <limits>
#include <mkl.h>
#include <vector>
#include <iomanip>
#include <boost/multiprecision/float128.hpp>
#include <mkl_spblas.h>
#include "ezini.h"
#include "helper.h"
#include "denseMatrix.h"
#include "sparseMatrix.h"
#include "my_blas.h"
#include "my_matmul.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>

using namespace std;

typedef boost::multiprecision::float128 float128;
const int maxNum = 16;

int main(int argc, char* argv[]) {
  int type, m_, n_, k_, batch_count, batch_size_, layout_ = 0, transA_ = 0, transB_ = 0, parallel_ = 0, incx_ = 1, incy_ = 1;
  int paramCount = 12;
  int totalParam = 0;
  FILE *fp;
  ini_entry_t entry[paramCount];

  fp = fopen("./config/config.ini", "r");

  //initialize entry structure to null
  for (int i = 0; i < paramCount; i++){
    entry[i].section = NULL;
    entry[i].key = NULL;
    entry[i].value = NULL;
  }
  for (int i = 0 ; i < paramCount; i++){
    if (GetEntryFromFile(fp, &entry[i]) > 0){
      totalParam++;
    }
  }
  fclose(fp);
  assert (totalParam >= 6);
  if (totalParam > 1) type = atoi(entry[0].value);               //obtain information required to run tests
  if (totalParam > 2) batch_count = atoi(entry[1].value);
  if (totalParam > 3) batch_size_ = atoi(entry[2].value);
  if (totalParam > 4) m_ = atoi(entry[3].value);
  if (totalParam > 5) n_ = atoi(entry[4].value);
  if (totalParam > 6) k_ = atoi(entry[5].value);
  if (totalParam > 7) layout_ = atoi(entry[6].value);
  if (totalParam > 8) transA_ = atoi(entry[7].value);
  if (totalParam > 9) transB_ = atoi(entry[8].value); 
  if (totalParam > 10) parallel_ = atoi(entry[9].value);
  if (totalParam > 11) incx_ = atoi(entry[10].value);
  if (totalParam >= 12) incy_ = atoi(entry[11].value);

  for (int i = 0; i < totalParam; i++){
    free(entry[i].section);
    free(entry[i].key);
    free(entry[i].value);
  }
  if (batch_count < 1){
    batch_count = 1;
  }
  if (batch_size_ < 1){
    batch_size_ = 1;
  }
  cout << "type: " << type << "\nbatch_count: " << batch_count << "\nbatch_size: " << batch_size_ << "\nm: " << m_ << "\nn: " << n_ << "\nk: " << k_ << "\nlayout: "<< layout_ << "\ntransA: " << transA_ << "\ntransB: " << transB_ << "\nparallel: " << parallel_ <<  "\nincx_: " << incx_ << "\nincy_: " << incy_ << endl;
  
  //this code segment with initializations and its functions were inspired from https://github.com/wudu98/fugaku_batch_gemm
  size_t align = 256;
  int *batch_size     = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));      //allocate memory for variables,  variables are dynamic arrays of size batch_count so each batch has its own parameters for computation
  int total_batch_size = 0;
    
  #pragma omp parallel for schedule(static) reduction(+ : total_batch_size) if (parallel_ == 1)         //allocate memory on different threads
  for (int i = 0; i < batch_count; i++){
    batch_size[i] = batch_size_;
    total_batch_size += batch_size[i];        //increment total batch size based on the batch size of every batch
  }
  CBLAS_LAYOUT layout = layout_ == 0 ? CblasRowMajor : CblasColMajor;       //set layout, transA, and transB according to parameters
  CBLAS_TRANSPOSE transA = transA_ == 0 ? CblasNoTrans : CblasTrans;
  CBLAS_TRANSPOSE transB = transB_ == 0 ? CblasNoTrans : CblasTrans;

  int *batch_head    = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));    //batch head indicate the start location of first matrix in each batch
  int *m      = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));
  int *n      = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));
  int *k      = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));
  int *lda    = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));
  int *ldb    = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));
  int *ldc    = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));
  int *incx   = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));
  int *incy   = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));
  int *lda_v  = static_cast<int*>(aligned_alloc(align, sizeof(int) * batch_count));
  batch_head[0] = 0;
  for (int i = 1; i < batch_count; i++){          //batch head corresponds to the start location of the first matrix for each batch
    batch_head[i] = batch_size[i-1] + batch_head[i-1];    //calculate the batch head for each batch by using the size and head of previous batch 
  }
  if (type == 0){ //half precision operations
    ProcessVariablesandPerformBLAS<MKL_F16>(parallel_, m, n, k, transA, transB, layout, lda, ldb, ldc, lda_v, incx, incy, incx_, incy_, m_, k_, n_, batch_head, batch_count, batch_size, total_batch_size, maxNum);
    performDenseMatrixOperations<MKL_F16>(m_, n_, k_, batch_size_, total_batch_size, parallel_, maxNum);
    performSparseMatrixOperations<MKL_F16>(m_, n_, k_, batch_size_, total_batch_size, parallel_, maxNum);
    ProcessandPerformSparseBLASOperations<MKL_F16>(0, total_batch_size, batch_size_, m_, n_, k_, transA_, layout_, maxNum, parallel_);
    ProcessandPerformSparseBLASOperations<MKL_F16>(1, total_batch_size, batch_size_, m_, n_, k_, transA_, layout_, maxNum, parallel_);  
  }
  else if (type == 1){  //single precision operations
    ProcessVariablesandPerformBLAS<float>(parallel_, m, n, k, transA, transB, layout, lda, ldb, ldc, lda_v, incx, incy, incx_, incy_, m_, k_, n_, batch_head, batch_count, batch_size, total_batch_size, maxNum);
    performDenseMatrixOperations<float>(m_, n_, k_, batch_size_, total_batch_size, parallel_, maxNum);
    performSparseMatrixOperations<float>(m_, n_, k_, batch_size_,  total_batch_size, parallel_, maxNum);
    ProcessandPerformSparseBLASOperations<float>(0, total_batch_size, batch_size_, m_, n_, k_, transA_, layout_, maxNum, parallel_);
    ProcessandPerformSparseBLASOperations<float>(1, total_batch_size, batch_size_, m_, n_, k_, transA_, layout_, maxNum, parallel_);
  }
  else if (type == 2){  //double precision operations
    ProcessVariablesandPerformBLAS<double>(parallel_, m, n, k, transA, transB, layout, lda, ldb, ldc, lda_v, incx, incy, incx_, incy_, m_, k_, n_, batch_head, batch_count, batch_size, total_batch_size, maxNum);
    performDenseMatrixOperations<double>(m_, n_, k_, batch_size_, total_batch_size, parallel_, maxNum);
    performSparseMatrixOperations<double>(m_, n_, k_, batch_size_, total_batch_size, parallel_, maxNum);
    ProcessandPerformSparseBLASOperations<double>(0, total_batch_size, batch_size_, m_, n_, k_, transA_, layout_, maxNum, parallel_);
    ProcessandPerformSparseBLASOperations<double>(1, total_batch_size, batch_size_, m_, n_, k_, transA_, layout_, maxNum, parallel_);
  }
  else if (type == 3){  //quad precision operations
    performDenseMatrixOperations<float128>(m_, n_, k_, batch_size_, total_batch_size, parallel_, maxNum);
    performSparseMatrixOperations<float128>(m_, n_, k_, batch_size_, total_batch_size, parallel_, maxNum);
    ProcessandPerformSparseBLASOperations<float128>(0, total_batch_size, batch_size_, m_, n_, k_, transA_, layout_, maxNum, parallel_);
    ProcessandPerformSparseBLASOperations<float128>(1, total_batch_size, batch_size_, m_, n_, k_, transA_, layout_, maxNum, parallel_);
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
  return 0;
}
