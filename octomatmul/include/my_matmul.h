template <typename T>
void performDenseMatrixOperations(const int validate, const int m_, const int n_, const int k_, const int batch_size, const int total_batch_size, const int parallel_, const int maxNum){
    DenseMatrix <T> ** a = new DenseMatrix<T>*[total_batch_size]; 
    DenseMatrix <T> ** b = new DenseMatrix<T>*[total_batch_size];
    DenseMatrix <T> ** c = new DenseMatrix<T>*[total_batch_size];
    vector < vector < T > > x;
    x.resize(total_batch_size);
    vector < vector < T > > y;
    y.resize(total_batch_size);

    for (int i = 0; i < total_batch_size; i++){
      a[i] = new DenseMatrix<T>(m_, k_, parallel_);
      b[i] = new DenseMatrix<T>(k_, n_, parallel_);
      c[i] = new DenseMatrix<T>();
      x[i].resize(k_);
      y[i].resize(k_);
      for (int j = 0; j < k_; j++){
        x[i][j] = getRandomValue(0, maxNum);
      }
    }
      for (int k = 0; k < total_batch_size; k++){
          for (int i = 0; i < m_; i++){
            for (int j = 0; j < k_; j++){
              a[k]->insert(i, j, getRandomValue<T>(0, maxNum));      //insert random elements to a and b 
          }
        }
      }
      for (int k = 0; k < total_batch_size; k++){
          for (int i = 0; i < k_; i++){
            for (int j = 0; j < n_; j++){
              b[k]->insert(i, j, getRandomValue<T>(0, maxNum));      //insert random elements to a and b 
            }
        }
      }
    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static) if (parallel_ == 1)
    for (int i = 0; i < total_batch_size; i++){
      *c[i] = (*a[i]) * (*b[i]);
    }
    double t1 = omp_get_wtime();

    cout << "Time for one matrix matrix multiplication operation with Dense Matrix: " << t1 - t0 << " ";
    obtainGflopsEFF<T>(t1-t0, total_batch_size, batch_size, m_, n_, k_);
    t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static) if (parallel_ == 1)
    for (int i = 0; i < total_batch_size; i++){
        y[i] = (*a[i]) * x[i];
    }
    t1 = omp_get_wtime();
    cout << "Time for one matrix vector multiplication operation with Dense Matrix: " << t1 - t0 << " ";
    obtainGflopsEFF<T>(t1-t0, total_batch_size, batch_size, m_, n_, k_);
    if (validate == 1){
      if (m_ == n_ && k_ == n_){
        if (ValidateDenseSparseMatrix<DenseMatrix, T>(a, b, c, m_, n_, k_, total_batch_size) == true){
          cout << "correct result\n\n";
        }
        else{
          cout << "incorrect result\n\n";
        }
      }
      else{
        cout << "Unable to validate, the dimensions of the matrix must equal\n"; 
      }
    }
    for (int i = 0; i < total_batch_size; i++) {
      delete a[i];
      delete b[i];
      delete c[i];
    }
    delete[] a;
    delete[] b;
    delete[] c;
}

template <typename T>
void performSparseMatrixOperations(const int validate, const int m_, const int n_, const int k_, const int batch_size, const int total_batch_size, const int parallel_, const int maxNum){
    SparseMatrix <T> ** a = new SparseMatrix<T>*[total_batch_size]; 
    SparseMatrix <T> ** b = new SparseMatrix<T>*[total_batch_size];
    SparseMatrix <T> ** c = new SparseMatrix<T>*[total_batch_size];
    vector< vector < T > > x;
    vector< vector < T > > y;
    x.resize(total_batch_size);
    y.resize(total_batch_size);
    for (int i = 0; i < total_batch_size; i++){         //initialize every element within the batch
      a[i] = new SparseMatrix<T>(m_, k_, rand() % ((m_*k_)/2), maxNum, parallel_);
      b[i] = new SparseMatrix<T>(k_, n_, rand() % ((k_*n_)/2), maxNum, parallel_);
      c[i] = new SparseMatrix<T>();
      x[i].resize(k_);
      y[i].resize(k_);
      for (int j = 0; j < k_; j++){
        x[i][j] = getRandomValue(0, maxNum);
      }
    }
    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static) if (parallel_ == 1)
    for (int i = 0; i < total_batch_size; i++){     //multiply every matrix within the batch arrays
      *c[i] = (*a[i]) * (*b[i]);
    }
    double t1 = omp_get_wtime();
    cout << "Time for one matrix multiplication operation with Sparse Matrix: " << t1 - t0 << " ";  
    obtainGflopsEFF<T>(t1-t0, total_batch_size, batch_size, m_, n_, k_);  
    t0 = omp_get_wtime();
    #pragma omp prallel for schedule(static) if (parallel_ == 1)
    for (int i = 0; i < total_batch_size; i++){
        y[i] = (*a[i]) * x[i];
    }
    t1 = omp_get_wtime();
    cout << "Time for one matrix vector multiplication operation with Sparse Matrix: " << t1 - t0 << " ";
    obtainGflopsEFF<T>(t1-t0, total_batch_size, batch_size, m_, n_, k_);
    if (validate == 1){
      if (m_ == n_ && k_ == n_){
        if (ValidateDenseSparseMatrix<SparseMatrix, T>(a, b, c, m_, n_, k_, total_batch_size) == true){
          cout << "correct result\n\n";
        }
        else{
          cout << "incorrect result\n\n";
        }
      }
      else{
        cout << "Unable to validate, the dimensions of the matrix must equal\n"; 
      }
    }
    for (int i = 0; i < total_batch_size; i++) {
      delete a[i];
      delete b[i];
      delete c[i];
    }
    delete[] a;
    delete[] b;
    delete[] c;
}

template <typename T>
void multiplyCSRSparseMatrices(const CSRMatrix<T>& A, const CSRMatrix<T>& B, CSRMatrix<T>& result, const int aRow, const int aCol, const int bRow, const int bCol, const int total_batch_size, const int parallel_) {
    int A_rows = aRow;
    int B_cols = bCol;
    for (int batch = 0; batch < total_batch_size; batch++) {
        result.values[batch] = new T[aRow * bCol];
        result.columns[batch] = new int[aRow * bCol];
        result.rowPtr[batch] = new int[A_rows + 1];
        result.nonZeros[batch] = 0;
        result.nonZeroCount[batch] = A_rows * B_cols; // initialize non zero count to the maximum possible number
        // Initialize the row pointer to zero
        result.rowPtr[batch][0] = 0;

        int* rowNonZeroCounts = new int[A_rows]();

        for (int i = 0; i < A_rows; ++i) {
            int rowStart = result.rowPtr[batch][i];
            for (int k = A.rowPtr[batch][i]; k < A.rowPtr[batch][i + 1]; ++k) {
              rowNonZeroCounts[i] = 0;
                int A_col = A.columns[batch][k];
                for (int l = B.rowPtr[batch][A_col]; l < B.rowPtr[batch][A_col + 1]; ++l) {
                    int j = B.columns[batch][l];
                    rowNonZeroCounts[i]++; 
                    result.values[batch][rowStart + rowNonZeroCounts[i]] = A.values[batch][k] * B.values[batch][l];
                    result.columns[batch][rowStart + rowNonZeroCounts[i]] = j;
                }
            }
            result.rowPtr[batch][i + 1] = rowStart + rowNonZeroCounts[i];
        }

        // Mark empty rows with rowPtr set to -1 for easy identification for later
        for (int i = 0; i < A_rows; ++i) {
            if (rowNonZeroCounts[i] == 0) {
                result.rowPtr[batch][i] = -1;
            }
        }

        delete[] rowNonZeroCounts;
    }

    // Fix rowPtr values for empty rows
    for (int batch = 0; batch < total_batch_size; batch++) {
        int prevRowPtr = 0;
        for (int i = 0; i < A_rows; ++i) {
            if (result.rowPtr[batch][i] == -1) {    
                result.rowPtr[batch][i] = prevRowPtr;
            } else {
                prevRowPtr = result.rowPtr[batch][i];
            }
        }
    }

    #pragma omp parallel for schedule(static) if (parallel_ == 1)
    for (int batch = 0; batch < total_batch_size; batch++) {
    int B_nonZeros = B.nonZeroCount[batch];
    int lastUp = 0; // position of the last update to the row index
    for (int i = 0; i < A_rows; ++i) {
        int count = result.rowPtr[batch][i]; // private count variable for each thread
        for (int j = 0; j < B_cols; ++j) {
            T sum = 0;
            for (int k = A.rowPtr[batch][i]; k < A.rowPtr[batch][i + 1]; ++k) {
                int A_col = A.columns[batch][k];
                for (int l = B.rowPtr[batch][A_col]; l < B.rowPtr[batch][A_col + 1]; ++l) {
                    if (B.columns[batch][l] == j) {
                        if constexpr (is_same_v<T, MKL_F16>) {
                            sum = f2h(h2f(sum) + h2f(A.values[batch][k]) * h2f(B.values[batch][l]));
                        } else {
                            sum += A.values[batch][k] * B.values[batch][l];
                        }
                        break;
                    }
                }
            }
            if (sum != 0) {
                result.values[batch][count] = sum; // append the result value if the sum isnt zero
                result.columns[batch][count] = j;
                count++;
            }
        }
        result.rowPtr[batch][i + 1] = count; // set last element as the number of non zeros
        result.nonZeros[batch] = count; // update non zeros count for this row
    }

        // resize array to fit the number of non-zero elements
        T* temp_values = new T[result.nonZeros[batch]];
        int* temp_columns = new int[result.nonZeros[batch]];

        for (int i = 0; i < result.nonZeros[batch]; ++i) {
            temp_values[i] = result.values[batch][i];
            temp_columns[i] = result.columns[batch][i];
        }

        delete[] result.values[batch];
        delete[] result.columns[batch];

        result.values[batch] = temp_values;
        result.columns[batch] = temp_columns;
    }
}
template <typename T>
void multiplyCSCSparseMatrices(const CSCMatrix<T>& A, const CSCMatrix<T>& B, CSCMatrix<T>& result, const int aRow, const int aCol, const int bRow, const int bCol, const int total_batch_size, const int parallel_) {
    int A_rows = aRow;
    int B_cols = bCol;
    for (int batch = 0; batch < total_batch_size; batch++){
      result.values[batch] = new T[aRow * bCol];
      result.rows[batch] = new int[aRow * bCol];
      result.colPtr[batch] = new int[B_cols + 1];
      result.colPtr[batch][0] = 0;
    }
    #pragma omp parallel for schedule(static) if (parallel_ == 1)
    for (int batch = 0; batch < total_batch_size; batch++) {
        int A_nonZeros = A.nonZeros[batch];
        int B_nonZeros = B.nonZeros[batch];
        int count = 0;
        for (int j = 0; j < B_cols; ++j){
            for (int k = 0; k < A_rows; ++k) {
                T sum = 0;
                for (int indexB = B.colPtr[batch][j]; indexB < B.colPtr[batch][j + 1]; ++indexB) {
                    int rowB = B.rows[batch][indexB];           //obtain row and value of B at the selected index within the column
                    T valueB = B.values[batch][indexB];

                    for (int indexA = A.colPtr[batch][rowB]; indexA < A.colPtr[batch][rowB + 1]; ++indexA) {
                        int colA = A.rows[batch][indexA];
                        if (colA == k) {
                            T valueA = A.values[batch][indexA];
                            if constexpr(is_same_v<T, MKL_F16>){
                              sum = f2h(h2f(sum) + h2f(valueA) * h2f(valueB));
                            }
                            else{
                              sum += valueA * valueB;
                            }
                            break;
                        }
                    }
                }

                if (sum != 0) {
                    result.values[batch][count] = sum;
                    result.rows[batch][count] = k;
                    ++count;
                }
            }
            result.colPtr[batch][j + 1] = count;
        }

        // Set the number of non-zero elements
        result.nonZeros[batch] = count;
        result.nonZeroCount[batch] = count;

        // Resize arrays to fit the number of non-zero elements
        T* temp_values = new T[count];
        int* temp_rows = new int[count];

        for (int i = 0; i < count; ++i) {
            temp_values[i] = result.values[batch][i];
            temp_rows[i] = result.rows[batch][i];
        }

        delete[] result.values[batch];
        delete[] result.rows[batch];

        result.values[batch] = temp_values;
        result.rows[batch] = temp_rows;
    }
}