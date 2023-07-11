#ifndef MY_BLAS_H
#define MY_BLAS_H

// #include <cblas.h>
#include <mkl.h>
#include <omp.h>
#include <type_traits>
#include <iomanip>

template <typename T>
void blas_batch_gemm(const int parallel, const int batch_count, const int* batch_size, const int* batch_head, const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB, const int* m, const int* n, const int* k, const T* alpha, const T* const * a, const int* lda, const T* const * b, const int* ldb, const T* beta, T** c, const int* ldc){
    int j = 0;
    #pragma omp parallel for schedule(static, 1) private(j) if (parallel == 1) 
    for (int i = 0; i < batch_count; i++){
        for (j = 0; j < batch_size[i]; j++){
            if constexpr(std::is_same_v<T, float>){
                cblas_sgemm(layout, transA, transB, m[i], n[i], k[i], alpha[i], a[batch_head[i]+j], lda[i], b[batch_head[i]+j], ldb[i], beta[i], c[batch_head[i]+j], ldc[i]);
            }
            else if constexpr(std::is_same_v<T, double>){
                cblas_dgemm(layout, transA, transB, m[i], n[i], k[i], alpha[i], a[batch_head[i]+j], lda[i], b[batch_head[i]+j], ldb[i], beta[i], c[batch_head[i]+j], ldc[i]);
            }
            else if constexpr(std::is_same_v<T, MKL_F16>){
                cblas_hgemm(layout, transA, transB, m[i], n[i], k[i], f2h(alpha[i]), a[batch_head[i]+j], lda[i], b[batch_head[i]+j], ldb[i], f2h(beta[i]), c[batch_head[i]+j], ldc[i]);
            }
        }
    }
}

template <typename T>
void blas_batch_gemv(const int parallel, const int batch_count, const int* batch_size, const int* batch_head, const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transA, const int* m, const int* n, const T* alpha, const T* const * a, const int* lda_v, const T* const * x, const int* incx, const T* beta, T** y, const int* incy){
    int j = 0;
      #pragma omp parallel for schedule(static, 1) private(j) if (parallel == 1) 
      for (int i = 0; i < batch_count; i++){
          for (j = 0; j < batch_size[i]; j++){
            if constexpr(std::is_same_v<T, float>){
                cblas_sgemv(layout,transA,m[i],n[i],alpha[i],a[batch_head[i]+j],lda_v[i],x[batch_head[i]+j],incx[i],beta[i],y[batch_head[i]+j],incy[i]);
            }
            else if constexpr(std::is_same_v<T, double>){
                cblas_dgemv(layout,transA,m[i],n[i],alpha[i],a[batch_head[i]+j],lda_v[i],x[batch_head[i]+j],incx[i],beta[i],y[batch_head[i]+j],incy[i]);
            }
          }
          
      }
}

template <typename T>
void ProcessVariablesandPerformBLAS(const int parallel_, int *m, int* n, int* k, const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB, const CBLAS_LAYOUT layout, int* lda, int* ldb, int* ldc, int* lda_v, int* incx, int* incy, int incx_, int incy_, int m_, int k_, int n_, const int* batch_head, const int batch_count, const int* batch_size, const int total_batch_size, const T maxNum){    
    size_t a_size, b_size, c_size, x_size, y_size, align=256;
    T* alpha = static_cast<T*>(aligned_alloc(align, sizeof(T)*batch_count));
    T* beta = static_cast<T*>(aligned_alloc(align, sizeof(T)*batch_count));
    T** a = static_cast<T**>(aligned_alloc(align, sizeof(T*)*total_batch_size));        // in a, b, c, x, and y, new matrices/vectors are stored at each row as 1d arrays. for example
    T** b = static_cast<T**>(aligned_alloc(align, sizeof(T*)*total_batch_size));        // a[0] would be the first matrix within the first batch, and a[0][0] would be the first element of that 
    T** c = static_cast<T**>(aligned_alloc(align, sizeof(T*)*total_batch_size));        // matrix, and the last matrix for the first batch would be a[batch_head[1]-1]
    T** x = static_cast<T**>(aligned_alloc(align, sizeof(T*)*total_batch_size));
    T** y = static_cast<T**>(aligned_alloc(align, sizeof(T*)*total_batch_size));
    for (int i = 0; i < batch_count; i++){      //initialize values within the array for different batches and call respective BLAS functions depending on variable type
      m[i] = m_;
      n[i] = n_;
      k[i] = k_;
      incx[i] = incx_;
      incy[i] = incy_;
      alpha[i] = 1.0;
      beta[i] = 0.0;
      lda_v[i] = layout == CblasColMajor ? m[i] : k[i]; 
      if ((transA == CblasNoTrans && layout == CblasColMajor) || (transA == CblasTrans && layout == CblasRowMajor)){
        lda[i] = m_;
        a_size = k[i] * lda[i];
      }
      else
      {
        lda[i] = k_;
        a_size = lda[i] * m[i];
      }
      if ((transB == CblasNoTrans && layout == CblasColMajor) || (transB == CblasTrans && layout == CblasRowMajor)){
        ldb[i] = k_;
        b_size = n[i] * ldb[i];
      }
      else{
        ldb[i] = n_;
        b_size = k[i] * ldb[i];
      }
      if (layout == CblasColMajor){
        ldc[i] = m_;
        c_size = n[i] * ldc[i];
      }
      else{
        ldc[i] = n_;
        c_size = m[i] * ldc[i];
      }
      x_size = transA == CblasNoTrans ? (1 + (k[i] - 1) * abs(incx_)) : (1 + (m[i] - 1) * abs(incx_));      //set the size of x and y arrays depending on whether or not utransposition is used 
      y_size = transA == CblasNoTrans ? (1 + (m[i] - 1) * abs(incy_)) : (1 + (k[i] - 1) * abs(incy_));
      for (int j = 0; j < batch_size[i]; j++){                          //allocate space for each matrix/vector and generate random numbers to fill them up
        a[batch_head[i]+j] = static_cast<T*>(aligned_alloc(align, sizeof(T) * a_size));
        b[batch_head[i]+j] = static_cast<T*>(aligned_alloc(align, sizeof(T) * b_size));
        c[batch_head[i]+j] = static_cast<T*>(aligned_alloc(align, sizeof(T) * c_size)); 
        x[batch_head[i]+j] = static_cast<T*>(aligned_alloc(align, sizeof(T) * x_size));
        y[batch_head[i]+j] = static_cast<T*>(aligned_alloc(align, sizeof(T) * y_size));
        for (int l = 0; l < a_size; l++){
          // a[batch_head[i]+j][l] = static_cast<T> (rand()) / static_cast<T> (RAND_MAX/maxNum);
          a[batch_head[i]+j][l] = getRandomValue<T>(0, maxNum);
        }
        for (int l = 0; l < b_size; l++){
          // b[batch_head[i]+j][l] = static_cast<T> (rand()) / static_cast<T> (RAND_MAX/maxNum);
          b[batch_head[i]+j][l] = getRandomValue<T>(0, maxNum);
        }
        for (int l = 0; l < c_size; l++){
          // c[batch_head[i]+j][l] = static_cast<T> (rand()) / static_cast<T> (RAND_MAX/maxNum);
          c[batch_head[i]+j][l] = getRandomValue<T>(0, maxNum);
        }
        for (int l = 0; l < x_size; l++){
          // x[batch_head[i]+j][l] = static_cast<T> (rand()) / static_cast<T> (RAND_MAX/maxNum);
          x[batch_head[i]+j][l] = getRandomValue<T>(0, maxNum);
        }
        for (int l = 0; l < y_size; l++){
          // y[batch_head[i]+j][l] = static_cast<T> (rand()) / static_cast<T> (RAND_MAX/maxNum);
          y[batch_head[i]+j][l] = getRandomValue<T>(0, maxNum);
        }
      }
    }
    if constexpr(std::is_same_v<T, float>){       //determine if T is float at compile time and run sgemm/sgemv
      double t0 = omp_get_wtime();
      blas_batch_gemm<float>(parallel_, batch_count, batch_size, batch_head, layout, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      double t1 = omp_get_wtime();
      cout << "time for one run of GEMM operation: " << t1 - t0 << endl;
      t0 = omp_get_wtime();
      blas_batch_gemv<float>(parallel_, batch_count, batch_size, batch_head, layout, transA, m, k, alpha, a, lda_v, x, incx, beta, y, incy);
      t1 = omp_get_wtime();
      cout << "time for one run of GEMV operation: " << t1 - t0 << endl;
    }
    else if constexpr(std::is_same_v<T, double>) {  //determine if T is double at compile time and run dgemm/dgemv
      double t0 = omp_get_wtime();
      blas_batch_gemm<double>(parallel_, batch_count, batch_size, batch_head, layout, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      double t1 = omp_get_wtime();
      cout << "time for one run of GEMM operation: " << t1 - t0 << endl;
      t0 = omp_get_wtime();
      blas_batch_gemv<double>(parallel_, batch_count, batch_size, batch_head, layout, transA, m, k, alpha, a, lda_v, x, incx, beta, y, incy);
      t1 = omp_get_wtime();
      cout << "time for one run of GEMV operation: " << t1 - t0 << endl;
    }
    else if constexpr(std::is_same_v<T, MKL_F16>){ //determine if T is MKL_F16 at compile time and run hgemm
      double t0 = omp_get_wtime();
      blas_batch_gemm<MKL_F16>(parallel_, batch_count, batch_size, batch_head, layout, transA, transB, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
      double t1 = omp_get_wtime();
      cout << "time for one run of GEMM operation: " << t1 - t0 << endl;
    }
    // for (int i = 0; i < 1; i++){
    //   cout << h2f(a[0][i]) << " a ";
    //   cout << h2f(b[0][i]) << " b ";
    //   cout << h2f(c[0][i]) << " c ";
    //   cout << endl;
    // }
    for (int i = 0; i < batch_count; i++){
      for (int j = 0; j < batch_size[i]; j++){
          free(a[batch_head[i]+j]);
          free(b[batch_head[i]+j]);
          free(c[batch_head[i]+j]);
          free(x[batch_head[i]+j]);
          free(y[batch_head[i]+j]);
      }
    }
    free(a);
    free(b);
    free(c);
    free(x);
    free(y);
    free(alpha);
    free(beta);
}
#endif  