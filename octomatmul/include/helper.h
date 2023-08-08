#ifndef RAND_GEN_H
#define RAND_GEN_H


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

template <typename T> // code taken from https://github.com/wudu98/fugaku_batch_gemm/blob/master/benchmark/batch_gemm_benchmark.c
double fp_peak(){ 
    int vlen = 64 / sizeof(T);
    int flop = vlen *  4;

    int ncore;
    #pragma omp parallel
    #pragma omp master
    ncore = omp_get_num_threads();

    double gFlops = 2.0 * ncore * flop;
    return gFlops;
}
template <typename T>
void performBLASGEMV(const int parallel, const int batch_count, const int* batch_size, const int* batch_head, const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transA, const int* m, const int* n, const T* alpha, const T* const * a, const int* lda_v, const T* const * x, const int* incx, const T* beta, T** y, const int* incy){
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
void obtainGflopsEFF(double time, int total_batch_size, int batch_size, int m, int n, int k){ // code taken from https://github.com/wudu98/fugaku_batch_gemm/blob/master/benchmark/batch_gemm_benchmark.c
  double gFlops = 2.0 * total_batch_size * batch_size * m * n * k / time * 1.e-9;
  double ratio = gFlops / fp_peak<T>();
  cout << "\n" << gFlops << " Gflops, at efficiency " << 100.*ratio << endl;
}

template <typename T>
T getRandomValue(T low, T high){
    static unsigned seed_val = time(NULL);      //set a fixed seed for testing purposes
    // cout << seed_val << endl;
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
struct CSCMatrix{
  T** values;
  int ** rows;
  int **colPtr;
  int * nonZeros;
  int * nonZeroCount;

  CSCMatrix(int total_batch_size){
    nonZeros = new int[total_batch_size];
    values = new T*[total_batch_size];
    colPtr = new int*[total_batch_size];
    rows = new int*[total_batch_size];
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
        int colCount = 0;
        for (int l = 0; l < n_; l++){
          //ensure that one column would not have too many values
            if ((matrix.nonZeroCount[i] < matrix.nonZeros[i]) && (colCount < (n_/2))  && (getRandomValue<double>(0.0, 1.0) < 0.5)){
              matrix.values[i][matrix.nonZeroCount[i]] = getRandomValue<T>(1, maxNum);      //generate random value, ensure that its a sparse matrix by generating a random double value betwween 0 to 1 and only taking anything below 0.5
              matrix.columns[i][matrix.nonZeroCount[i]] = l;
              matrix.nonZeroCount[i]++;
              colCount++;
          }
        }
        //check to make sure every element has at least one value
        if (colCount == 0){
          matrix.values[i][matrix.nonZeroCount[i]] = getRandomValue<T>(1, maxNum);
          matrix.columns[i][matrix.nonZeroCount[i]] = getRandomValue<int>(0, n_-1);
          matrix.nonZeroCount[i]++;
        }
        matrix.rowPtr[i][j+1] = matrix.nonZeroCount[i];
      }
      matrix.rowPtr[i][m_] = matrix.nonZeroCount[i];    //set last element of rowPtr to be the number of non zeros
      matrix.nonZeros[i] = matrix.nonZeroCount[i];
   }
}
template <typename T>
void ProcessCSCMatrix(CSCMatrix<T> &matrix, const int total_batch_size, const int m_, const int n_, const int maxNum){
  for (int i = 0; i < total_batch_size; i++){
      matrix.nonZeros[i] = (m_*n_) / 2;
      matrix.values[i] = new T[matrix.nonZeros[i]];
      matrix.rows[i] = new int[matrix.nonZeros[i]];
      matrix.colPtr[i] = new int[n_+1];
      matrix.colPtr[i][0] = 0;
      matrix.nonZeroCount[i] = 0;
      for (int j = 0; j < n_; j++){
        int rowCount = 0;
        for (int l = 0; l < m_; l++){
          if ((matrix.nonZeroCount[i] < matrix.nonZeros[i]) && (rowCount < m_/2) && (getRandomValue<double>(0.0, 1.0) < 0.5)){
            matrix.values[i][matrix.nonZeroCount[i]] = getRandomValue<T>(1, maxNum);
            matrix.rows[i][matrix.nonZeroCount[i]] = l;
            matrix.nonZeroCount[i]++;
            rowCount++;
          }
        }
        if (rowCount == 0){
          matrix.values[i][matrix.nonZeroCount[i]] = getRandomValue<T>(1, maxNum);
          matrix.rows[i][matrix.nonZeroCount[i]] = getRandomValue<int>(0, m_-1);
          matrix.nonZeroCount[i]++;
        }
        matrix.colPtr[i][j+1] = matrix.nonZeroCount[i];
      }
      matrix.colPtr[i][n_] = matrix.nonZeroCount[i];    //set last element of Ptr to be the number of non zeros
      matrix.nonZeros[i] = matrix.nonZeroCount[i];
  }
}
template <typename T>
void obtainCSRandPrint(sparse_matrix_t mat, int m, int n){
    MKL_INT * rows_start, *rows_end, *col_indx;
    sparse_index_base_t index = SPARSE_INDEX_BASE_ZERO;
    T * values;
    if constexpr(is_same_v<T, float>){
      mkl_sparse_s_export_csr(mat, &index, &m, &n, &rows_start, &rows_end, &col_indx, &values);
    }
    else if constexpr(is_same_v<T, double>){
      mkl_sparse_d_export_csr(mat, &index, &m, &n, &rows_start, &rows_end, &col_indx, &values);
    }
    T denseC[m * n];
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
            cout << denseC[i * n + j] << " ";
        }
        cout << endl;;
    }
    cout << endl;
}

template <typename T>
void obtainCSCandPrint(sparse_matrix_t mat, int m, int n) {
  MKL_INT *col_start, *col_end, *row_indx;
  sparse_index_base_t index = SPARSE_INDEX_BASE_ZERO;
  T *values;
  if constexpr(is_same_v<T, float>){
    mkl_sparse_s_export_csc(mat, &index, &m, &n, &col_start, &col_end, &row_indx, &values);
  }
  else if constexpr(is_same_v<T, double>){
    mkl_sparse_d_export_csc(mat, &index, &m, &n, &col_start, &col_end, &row_indx, &values);
  }
  
  T denseC[m * n];
  for (MKL_INT i = 0; i < m * n; i++) {
      denseC[i] = 0.0;  // initialize element to 0
  }

  for (MKL_INT j = 0; j < n; j++) {     // convert sparse matrix to dense matrix for output purposes when testing
      for (MKL_INT i = col_start[j]; i < col_end[j]; i++) {
          denseC[row_indx[i] * n + j] = values[i];
      }
  }

  for (MKL_INT i = 0; i < m; i++) {     // print matrix 
      for (MKL_INT j = 0; j < n; j++) {
          cout << denseC[i * n + j] << " ";
      }
      cout << endl;
  }
  cout << endl;
}

template <typename T>
void generateVector(T* &x, const int size, const int maxNum){
  for (int i = 0; i < size; i++){
    x[i] = getRandomValue<T>(0, maxNum);
  }
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
template <typename T>
void PrintCSRMatrix(const CSRMatrix<T> m1,  const int m_, const int n_){
  //m1.rowPtr[i] = start of current row
  //m1.columns[rowPtr[i] + curCol] = at that row, the column corresponding to the one we're currently on. For instance, row = 2, rowPtr[i] points to the section in columns that corresponds to row 2, and then + curCol allows you to traverse the column
  //m1.values[rowPtr[i] + curCol] = value at that row and col
  for (int j = 0; j < m_ ; j++){
    for (int i = m1.rowPtr[0][j]; i < m1.rowPtr[0][j+1]; i++){
      cout << "row start at: " <<  m1.rowPtr[0][j] <<  " row: " << j << " value: " << m1.values[0][i] << "  columns: " << m1.columns[0][i] << endl;
    }
  }
  cout << "Matrix: " << endl;
  for (int i = 0; i < m_; i++){
    int curCol = 0;
    for (int j = 0; j < n_; j++){
      if (m1.columns[0][m1.rowPtr[0][i] + curCol] == j && m1.rowPtr[0][i] + curCol < m1.rowPtr[0][i+1]){
        if constexpr(is_same_v<T, MKL_F16>){
          cout <<  h2f(m1.values[0][m1.rowPtr[0][i]+curCol]) << " "; 
        }
        else{
          cout <<  m1.values[0][m1.rowPtr[0][i]+curCol] << " ";
        }
        curCol++;
      }
      else{
        cout << "0" << " ";
      }
    }
    cout << endl;
  }
}
template <typename T>
void PrintCSCMatrix(const CSCMatrix<T>& m1, const int m_, const int n_) {
    // for (int j = 0; j < n_ ; j++){
    //   for (int i = m1.colPtr[0][j]; i < m1.colPtr[0][j+1]; i++){
    //     cout << "col start at: " <<  m1.colPtr[0][j] <<  " col: " << j << " value: " << h2f(m1.values[0][i]) << "  rows: " << m1.rows[0][i] << endl;
    //   }
    // }
    cout << "Matrix:" << endl;
    int count = 0;
    for (int i = 0; i < m_; i++) {
        for (int j = 0; j < n_; j++) {
            bool found = false;
            for (int k = m1.colPtr[0][j]; k < m1.colPtr[0][j + 1]; k++) {
                if (m1.rows[0][k] == i) {
                    if constexpr(is_same_v<T, MKL_F16>){
                      cout << h2f(m1.values[0][k]) << " ";
                    }
                    else{
                      cout << m1.values[0][k] << " ";
                    }
                    found = true;
                    break;
                }
            }
            if (!found) {
                cout << "0 ";
            }
        }
        cout << endl;
    }
}

template <typename T>
void transposeCSRMatrix(const CSRMatrix<T>& m1, CSRMatrix<T>& result, const int m, const int n, int total_batch_size){
    for (int i = 0; i < total_batch_size; i++){
        vector<int> rowPtr(n+2, 0);
        int resRow = n;
        int resCol = m;
        result.nonZeros[i] = m1.nonZeros[i];
        result.nonZeroCount[i] = m1.nonZeroCount[i];
        result.rowPtr[i] = new int[resRow+1];
        result.columns[i] = new int[result.nonZeros[i]];
        result.values[i] = new T[result.nonZeros[i]];
        //initialize rowPtr for all rows that would be covered
        for (int j = 0; j < m1.nonZeroCount[i]; j++){                       //code logic taken from https://stackoverflow.com/questions/49395986/compressed-sparse-row-transpose
            ++rowPtr[m1.columns[i][j] + 2];
        } 

        for (int j = 2; j < rowPtr.size(); j++){
            rowPtr[j] += rowPtr[j-1];
        }   
        //loop through original m1 matrix and transpose by swapping row and column values
        for (int j = 0; j < m; j++){
            for (int k = m1.rowPtr[i][j]; k < m1.rowPtr[i][j+1]; k++){          
                int index = rowPtr[m1.columns[i][k] + 1]++;
                T val = m1.values[i][k];
                result.values[i][index] = val;
                result.columns[i][index] = j;
            }
        }
      for (int j = 0; j < resRow+1; j++){
        result.rowPtr[i][j] = rowPtr[j];
      }
      rowPtr.clear();
    }
}
template <typename T>
void transposeCSCMatrix(const CSCMatrix<T>& m1, CSCMatrix<T>& result, const int m, const int n, const int total_batch_size) {
    for (int i = 0; i < total_batch_size; i++) {
        int resRow = n;
        int resCol = m;
        result.nonZeros[i] = m1.nonZeros[i];
        result.nonZeroCount[i] = m1.nonZeroCount[i];
        result.colPtr[i] = new int[resCol + 1];
        result.rows[i] = new int[result.nonZeros[i]];
        result.values[i] = new T[result.nonZeros[i]];
        
        for (int j = 0; j < resCol+1; j++){
             result.colPtr[i][j] = 0;
        }
        // initialize colPtr for all columns that would be covered
        for (int j = 0; j < m1.nonZeros[i]; j++) {
            ++result.colPtr[i][m1.rows[i][j] + 1];
        }
        for (int j = 1; j < resCol + 1; j++) {
            result.colPtr[i][j] += result.colPtr[i][j - 1];
        }

        // temp array to keep track of number of elemennts added to each col
        int colCount[resCol];
        std::fill(colCount, colCount + resCol, 0);

        // loop through the original m1 matrix and transpose
        for (int j = 0; j < n; j++) {
            for (int k = m1.colPtr[i][j]; k < m1.colPtr[i][j + 1]; k++) {
                int row = m1.rows[i][k];
                int index = result.colPtr[i][row] + colCount[row]++;
                T val = m1.values[i][k];
                result.values[i][index] = val;
                result.rows[i][index] = j;
            }
        }
    }
}
template <typename T>
bool ValidateBLASMatrix(const int parallel, const CBLAS_LAYOUT layout, const CBLAS_TRANSPOSE transA, const CBLAS_TRANSPOSE transB, const T* const* a, const T* const* b, const T* const* c, const int* m, const int* n, const int* k, const int m_, const int n_, const int k_, const int batch_count, const int* batch_size, const int* batch_head, const int total_batch_size, const int maxNum, const T* alpha, const T* beta, const int* lda_v, const int* incx, const int* incy){
    bool isEqual = true;            //employs Freivald's algorithm to validate the results of matrix multiplication
    CBLAS_TRANSPOSE transC = CblasNoTrans;
    if (m_ == n_ && n_ == k_){
      size_t align = 256;
      T** x = static_cast<T**>(aligned_alloc(align, sizeof(T*)*total_batch_size));      //allocate for arrays to store result
      T** y = static_cast<T**>(aligned_alloc(align, sizeof(T*)*total_batch_size));    
      T** z = static_cast<T**>(aligned_alloc(align, sizeof(T*)*total_batch_size));
      T** r = static_cast<T**>(aligned_alloc(align, sizeof(T*)*total_batch_size));
      for (int i = 0; i < batch_count; i++){
        for (int j = 0; j < batch_size[i]; j++){
          x[batch_head[i]+j] = static_cast<T*>(aligned_alloc(align, sizeof(T) * n_));
          y[batch_head[i]+j] = static_cast<T*>(aligned_alloc(align, sizeof(T) * n_));
          z[batch_head[i]+j] = static_cast<T*>(aligned_alloc(align, sizeof(T) * n_));
          r[batch_head[i]+j] = static_cast<T*>(aligned_alloc(align, sizeof(T) * n_));
          for (int v = 0; v < n_; v++) {
            if (getRandomValue<double>(0.0, 1.0) >= 0.50) {
                  if constexpr (is_same_v<T, MKL_F16>) {        //randomly generate array elements either 0 or 1 
                      x[batch_head[i]+j][v] = f2h(1);
                  }
                  else {
                      x[batch_head[i]+j][v] = 1;
                  }
              } 
              else {
                  if constexpr (is_same_v<T, MKL_F16>) {
                      x[batch_head[i]+j][v] = f2h(0);
                  } 
                  else {
                      x[batch_head[i]+j][v] = 0;
                  }
              }
            }
        }
      }
      performBLASGEMV<T>(parallel, batch_count, batch_size, batch_head, layout, transC, m, k, alpha, c, lda_v, x, incx, beta, y, incy);     //compute CX stored into Y, BX stored into Z, and then AZ stored into R. Then we compare Y == Z to determine if the matrix multiplication result is correct
      performBLASGEMV<T>(parallel, batch_count, batch_size, batch_head, layout, transB, m, k, alpha, b, lda_v, x, incx, beta, z, incy);
      performBLASGEMV<T>(parallel, batch_count, batch_size, batch_head, layout, transA, m, k, alpha, a, lda_v, z, incx, beta, r, incy);

      for (int i = 0; i < batch_count; i++){
        for (int j = 0; j < batch_size[i]; j++){
          if (isEqual == true){
            for (int v = 0; v < n_; v++){
                if ((r[batch_head[i]+j][v] - y[batch_head[i]+j][v]) < 0.001 != true){
                  isEqual = false;
                  break;
              }
          }
        }
        else{
            break;
        }
      }
    }
    for (int i = 0; i < batch_count; i++){        //free memory
      for (int j = 0; j < batch_size[i]; j++){
          free(x[batch_head[i]+j]);
          free(y[batch_head[i]+j]);
          free(z[batch_head[i]+j]);
          free(r[batch_head[i]+j]);
      }
    }
    free(x);
    free(y);
    free(z);
    free(r);
    return isEqual;
  }
  return false;

}
template <template <typename> class Matrix, typename T>
bool ValidateMyCSRCSCMatrix(const Matrix<T>& A, const Matrix<T>& B, const Matrix<T>& C, int m, int n, int k, int total_batch_size, int parallel){
    bool isEqual = true;        //also employs the Freivald's algorithm
    if (m == n && n == k){
        T **x = new T*[total_batch_size];
        T **y = new T*[total_batch_size];
        T **z = new T*[total_batch_size];
        T **r = new T*[total_batch_size];
        for (int i = 0; i < total_batch_size; i++){
            x[i] = new T[n];
            y[i] = new T[n];
            z[i] = new T[n];
            r[i] = new T[n];
            generateVector(y[i], n, 0);
            generateVector(z[i], n, 0);
            generateVector(r[i], n, 0);
            for (int j = 0; j < n; j++){
                if (getRandomValue<double>(0.0, 1.0) >= 0.50){
                    if constexpr(is_same_v<T, MKL_F16>){
                        x[i][j] = f2h(1);
                    }
                    else{
                        x[i][j] = 1;
                    }
                }
                else{
                    if constexpr(is_same_v<T, MKL_F16>){
                        x[i][j] = f2h(0);
                    }
                    else{
                        x[i][j] = 0;
                    }
                }
            }
        }
        if constexpr(is_same_v<Matrix<T>, CSRMatrix<T>>){
            CSRMVMultiply(C,x,y,total_batch_size, n, n, parallel);
            CSRMVMultiply(B,x,z,total_batch_size, n, n, parallel);
            CSRMVMultiply(A,z,r,total_batch_size, n, n, parallel);
        }
        else if constexpr(is_same_v<Matrix<T>, CSCMatrix<T>>){
            CSCMVMultiply(C,x,y,total_batch_size, n, n, parallel);
            CSCMVMultiply(B,x,z,total_batch_size, n, n, parallel);
            CSCMVMultiply(A,z,r,total_batch_size, n, n, parallel);
            
        }
        for (int j = 0; j < total_batch_size; j++){
            if (isEqual == true){
                for (int k = 0; k < n; k++){
                    if constexpr(is_same_v<T, MKL_F16>){
                        if (std::abs((h2f(r[j][k]) - h2f(y[j][k])) <= 5) != true){
                            isEqual = false;
                            break;
                        }
                    }
                    else{
                        if ((r[j][k] - y[j][k]) < 0.001 != true){
                            isEqual = false;
                            break;
                        }
                    }
                }
            }
            else{
                break;
            }
        }
    for (int i = 0; i < total_batch_size; i++){
        delete [] x[i];
        delete [] y[i]; 
        delete [] z[i];
        delete [] r[i];
    }
    delete [] x;
    delete [] y;
    delete [] z;
    delete [] r;
    return isEqual;
}
    return false;
}
template <template <typename> class Matrix, typename T>
bool ValidateDenseSparseMatrix(Matrix<T>** A, Matrix<T>**  B, Matrix<T>**  C, int m, int n, int k, int total_batch_size){
    bool isEqual = true;        //also employs the Freivvald's algorithm
    if (m == n && n == k){
        vector<vector<T>> x(total_batch_size, vector<T>(n, 0));
        vector<vector<T>> y(total_batch_size, vector<T>(n, 0));
        vector<vector<T>> z(total_batch_size, vector<T>(n, 0));
        vector<vector<T>> r(total_batch_size, vector<T>(n, 0));
        for (int i = 0; i < total_batch_size; i++) {
            x[i].resize(n, 0); // Resize x[i] to have 'n' elements
            for (int j = 0; j < n; j++) {
                if (getRandomValue<double>(0.0, 1.0) >= 0.50) {
                    if constexpr (is_same_v<T, MKL_F16>) {
                        x[i][j] = f2h(1);
                    } else {
                        x[i][j] = 1;
                    }
                } else {
                    if constexpr (is_same_v<T, MKL_F16>) {
                        x[i][j] = f2h(0);
                    } else {
                        x[i][j] = 0;
                    }
                }
            }
        }
            for (int i = 0; i < total_batch_size; i++){
                z[i] = (*B[i]) * x[i];
            }
            for (int i = 0; i < total_batch_size; i++){
                y[i] = (*C[i]) * x[i];
            }
            for (int i = 0; i < total_batch_size; i++){
                r[i] = (*A[i]) * z[i];
            }
            for (int j = 0; j < total_batch_size; j++){
                if (isEqual == true){
                    for (int k = 0; k < n; k++){
                        if constexpr(is_same_v<T, MKL_F16>){
                            if (std::abs((h2f(r[j][k]) - h2f(y[j][k])) <= 5) != true){
                                isEqual = false;
                                break;
                            }
                        }
                        else{
                            if (r[j][k] - y[j][k] < 0.001 != true){
                                isEqual = false;
                                break;
                            }
                        }
                    }
                }
                else{
                    break;
                }
           } 
        return isEqual;  
    }
    return false;
}
#endif