#ifndef RAND_GEN_H
#define RAND_GEN_H

// #include <random>
// #include <type_traits>
// #include <limits>
// #include <mkl.h>

using namespace std;
                  //below code segment taken from common_func.c which comes with the installation of Intel oneAPI Math Kernel Library for use with cblas_hgemm() function
typedef union {       
  MKL_F16 raw;
  struct {
    unsigned int frac : 10;
    unsigned int exp  :  5;
    unsigned int sign :  1;
  } bits;
} conv_union_f16; 

typedef union {
  float raw;
  struct {
    unsigned int frac : 23;
    unsigned int exp  :  8;
    unsigned int sign :  1;
  } bits;
} conv_union_f32;

static float h2f(MKL_F16 x) {       //convert half precision to f16, taken from common_func.c
  conv_union_f16 src;
  conv_union_f32 dst;

  src.raw = x;
  dst.raw = 0;
  dst.bits.sign = src.bits.sign;

  if (src.bits.exp == 0x01f) {
    dst.bits.exp = 0xff;
    if (src.bits.frac > 0) {
      dst.bits.frac = ((src.bits.frac | 0x200) << 13);
    }
  } else if (src.bits.exp > 0x00) {
    dst.bits.exp = src.bits.exp + ((1 << 7) - (1 << 4));
    dst.bits.frac = (src.bits.frac << 13);
  } else {
    unsigned int v = (src.bits.frac << 13);

    if (v > 0) {
      dst.bits.exp = 0x71;
      while ((v & 0x800000UL) == 0) {
        dst.bits.exp--;
        v <<= 1;
      }
      dst.bits.frac = v;
    }
  }

  return dst.raw;
}

static MKL_F16 f2h(float x) { //convert f16 to half precision, taken from common_func.c
  conv_union_f32 src;
  conv_union_f16 dst;

  src.raw = x;
  dst.raw = 0;
  dst.bits.sign = src.bits.sign;

  if (src.bits.exp == 0x0ff) {
    dst.bits.exp = 0x01f;
    dst.bits.frac = (src.bits.frac >> 13);
    if (src.bits.frac > 0) {
      dst.bits.frac |= 0x200;
    }
  } else if (src.bits.exp >= 0x08f) {
    dst.bits.exp = 0x01f;
    dst.bits.frac = 0x000;
  } else if (src.bits.exp >= 0x071) {
    dst.bits.exp = src.bits.exp + ((1 << 4) - (1 << 7));
    dst.bits.frac = (src.bits.frac >> 13);
  } else if (src.bits.exp >= 0x067) {
    dst.bits.exp = 0x000;
    if (src.bits.frac > 0) {
      dst.bits.frac = (((1U << 23) | src.bits.frac) >> 14);
    } else {
      dst.bits.frac = 1;
    }
  }

  return dst.raw;
}

template <typename T>
T getRandomValue(T low, T high){
    static unsigned seed_val = 15;      //set a fixed seed for testing purposes
    static mt19937 gen(seed_val); 
    if constexpr(std::is_same_v<T, MKL_F16>){     //generate random float values and convert them to mkl_f16 using f2h
        std::uniform_real_distribution<float> dis(static_cast<float>(low), static_cast<float>(high));
        return f2h(dis(gen));
    }
    else if constexpr (std::is_same_v<T, boost::multiprecision::float128>){
      std::uniform_real_distribution<long double> dis(static_cast<long double>(low), static_cast<long double>(high));
      return static_cast<boost::multiprecision::float128>(dis(gen));
    }
    else if constexpr (std::is_integral<T>::value) {      //generate int values
        std::uniform_int_distribution<T> dis(low, high);
        return dis(gen);
    }
    else if constexpr (std::is_floating_point<T>::value) {    //generate fp values
        std::uniform_real_distribution<T> dis(low, high);
        return dis(gen);
    }
    return T(0.0);
}
template <typename T>
struct CSRMatrix{
  T** values;
  int ** columns;
  int ** rowPtr;
  int * nonZeros;
  int * nonZeroCount;

  CSRMatrix(int total_batch_size){
    nonZeros = new int[total_batch_size];        //CSR format with values, column, and row pointer
    values = new T*[total_batch_size];
    columns = new int*[total_batch_size];
    rowPtr = new int*[total_batch_size];
    nonZeroCount = new int[total_batch_size];
  }
};

template <typename T>
void ProcessCSRMatrix(CSRMatrix<T> &matrix, const int total_batch_size, const int m_, const int n_, const int maxNum){
   for (int i = 0; i < total_batch_size; i++){
      matrix.nonZeros[i] = (m_*n_) / 2;
      matrix.values[i] = new T[matrix.nonZeros[i]];
      matrix.columns[i] = new int[matrix.nonZeros[i]];
      matrix.rowPtr[i] = new int[m_+1];
      matrix.rowPtr[i][0] = 0;
      matrix.nonZeroCount[i] = 0;
      for (int j = 0; j < m_; j++){
        for (int l = 0; l < n_; l++){
            if ((matrix.nonZeroCount[i] < matrix.nonZeros[i]) && (getRandomValue<double>(0.0, 1.0) < 0.5)){
              matrix.values[i][matrix.nonZeroCount[i]] = getRandomValue<T>(0, maxNum);      //generate random value, ensure that its a sparse matrix by generating a random double value betwween 0 to 1 and only taking anything below 0.5
              matrix.columns[i][matrix.nonZeroCount[i]] = l;
              matrix.nonZeroCount[i]++;
          }
        }
        matrix.rowPtr[i][j+1] = matrix.nonZeroCount[i];
      }
   }
}
template <typename T>
void obtainCSRandPrint(sparse_matrix_t mat, int m, int n){
  MKL_INT * rows_start, *rows_end, *col_indx;
  sparse_index_base_t index = SPARSE_INDEX_BASE_ZERO;
  T * values;
  mkl_sparse_s_export_csr(mat, &index, &m, &n, &rows_start, &rows_end, &col_indx, &values);
    double denseC[m * n];
    for (MKL_INT i = 0; i < m * n; i++) {
        denseC[i] = 0.0;  // initialize element to 0
    }

    
    for (MKL_INT i = 0; i < m; i++) {     //convert sparse matrix to dense matrix for output purposes when testing
        for (MKL_INT j = rows_start[i]; j < rows_end[i]; j++) {
            denseC[i * n + col_indx[j]] = values[j];
        }
    }

    for (MKL_INT i = 0; i < m; i++) {       //print matrix 
        for (MKL_INT j = 0; j < n; j++) {
            printf("%f ", denseC[i * n + j]);
        }
        printf("\n");
    }
  cout << endl;
}
template <typename T>
void generateDenseMatrix(const int layout, T* &matrix, const int rows, const int columns, const int maxNum)
{
  if (layout == 0){     //if row major
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            matrix[i * columns + j] = getRandomValue<T>(0, maxNum);
        }
    }
  }
  else if (layout == 1){      //if column major
    for (int j = 0; j < columns; ++j) {
        for (int i = 0; i < rows; ++i) {
            matrix[i + j * rows] = getRandomValue<T>(0, maxNum);
        }
    }
  }
}
#endif