#ifndef RAND_GEN_H
#define RAND_GEN_H


typedef boost::multiprecision::float128 float128;
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
    static unsigned seed_val = time(NULL);      //set a fixed seed for testing purposes
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
  // for (int i = 0; i < matrix.nonZeros[0]; i++){
  //     cout << matrix.values[0][i] << " ";
  // }
  // cout << endl;
  // for (int i = 0; i < matrix.nonZeros[0]; i++){
  //     cout << matrix.columns[0][i] << " ";
  // }
  //  cout << endl;
  //  for (int i = 0; i < m_+1; i++){
  //     cout << matrix.rowPtr[0][i] << " ";
  //  }
  //  cout << endl;
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
  // for (int i = 0; i < matrix.nonZeros[0]; i++){
  //     cout << matrix.values[0][i] << " ";
  // }
  // cout << endl;
  // for (int i = 0; i < matrix.nonZeros[0]; i++){
  //     cout << matrix.rows[0][i] << " ";
  // }
  //  cout << endl;
  //  for (int i = 0; i < n_+1; i++){
  //     cout << matrix.colPtr[0][i] << " ";
  //  }
  //  cout << endl;
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
// template <typename T>
// void PrintCSRMatrix(const CSRMatrix<T> m1,  const int m_, const int n_){
//   //m1.rowPtr[i] = start of current row
//   //m1.columns[rowPtr[i] + curCol] = at that row, the column corresponding to the one we're currently on. For instance, row = 2, rowPtr[i] points to the section in columns that corresponds to row 2, and then + curCol allows you to traverse the column
//   //m1.values[rowPtr[i] + curCol] = value at that row and col
//   // for (int j = 0; j < m_ ; j++){
//   //   for (int i = m1.rowPtr[0][j]; i < m1.rowPtr[0][j+1]; i++){
//   //     cout << "row start at: " <<  m1.rowPtr[0][j] <<  " row: " << j << " value: " << m1.values[0][i] << "  columns: " << m1.columns[0][i] << endl;
//   //   }
//   // }
//   cout << "Matrix: " << endl;
//   for (int i = 0; i < m_; i++){
//     int curCol = 0;
//     for (int j = 0; j < n_; j++){
//       if (m1.columns[0][m1.rowPtr[0][i] + curCol] == j && m1.rowPtr[0][i] + curCol < m1.rowPtr[0][i+1]){
//         cout <<  (m1.values[0][m1.rowPtr[0][i]+curCol]) << " "; 
//         curCol++;
//       }
//       else{
//         cout << "0" << " ";
//       }
//     }
//     cout << endl;
//   }
// }
// template <typename T>
// void PrintCSCMatrix(const CSCMatrix<T>& m1, const int m_, const int n_) {
//     // for (int j = 0; j < n_ ; j++){
//     //   for (int i = m1.colPtr[0][j]; i < m1.colPtr[0][j+1]; i++){
//     //     cout << "col start at: " <<  m1.colPtr[0][j] <<  " col: " << j << " value: " << m1.values[0][i] << "  rows: " << m1.rows[0][i] << endl;
//     //   }
//     // }
//     cout << "Matrix:" << endl;
//     int count = 0;
//     for (int i = 0; i < m_; i++) {
//         for (int j = 0; j < n_; j++) {
//             bool found = false;
//             for (int k = m1.colPtr[0][j]; k < m1.colPtr[0][j + 1]; k++) {
//                 if (m1.rows[0][k] == i) {
//                     cout << (m1.values[0][k]) << " ";
//                     found = true;
//                     break;
//                 }
//             }
//             if (!found) {
//                 cout << "0 ";
//             }
//         }
//         cout << endl;
//     }
// }

template <typename T>
CSRMatrix<T> transposeCSRMatrix(const CSRMatrix<T>& m1, const int m, const int n, int total_batch_size){
    CSRMatrix<T> result(total_batch_size);
    for (int i = 0; i < total_batch_size; i++){
        int resRow = n;
        int resCol = m;
        result.nonZeros[i] = m1.nonZeros[i];
        result.nonZeroCount[i] = m1.nonZeroCount[i];
        result.rowPtr[i] = new int[resRow+1];
        result.columns[i] = new int[result.nonZeros[i]];
        result.values[i] = new T[result.nonZeros[i]];
        //initialize rowPtr for all rows that would be covered
        for (int j = 0; j < m1.nonZeros[i]; j++){                       //code logic taken from https://stackoverflow.com/questions/49395986/compressed-sparse-row-transpose
            ++result.rowPtr[i][m1.columns[i][j] + 2];
        } 

        for (int j = 1; j < resRow+1; j++){
            result.rowPtr[i][j] += result.rowPtr[i][j-1];
        }   
        //loop through original m1 matrix and transpose by swapping row and column values
        for (int j = 0; j < m; j++){
            for (int k = m1.rowPtr[i][j]; k < m1.rowPtr[i][j+1]; k++){          
                int index = result.rowPtr[i][m1.columns[i][k] + 1]++;
                T val = m1.values[i][k];
                result.values[i][index] = val;
                result.columns[i][index] = j;
            }
        }

    }
    return result;
}
template <typename T>
CSCMatrix<T> transposeCSCMatrix(const CSCMatrix<T>& m1, const int m, const int n, const int total_batch_size) {
    CSCMatrix<T> result(total_batch_size);
    for (int i = 0; i < total_batch_size; i++) {
        int resRow = n;
        int resCol = m;
        result.nonZeros[i] = m1.nonZeros[i];
        result.nonZeroCount[i] = m1.nonZeroCount[i];
        result.colPtr[i] = new int[resCol + 1];
        result.rows[i] = new int[result.nonZeros[i]];
        result.values[i] = new T[result.nonZeros[i]];
        
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
    return result;
}
#endif