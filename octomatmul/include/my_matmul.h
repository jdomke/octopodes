template <typename T>
void performDenseMatrixOperations(const int m_, const int n_, const int k_, const int total_batch_size, const int parallel_, const int maxNum){
    DenseMatrix <T> ** a = new DenseMatrix<T>*[total_batch_size]; 
    DenseMatrix <T> ** b = new DenseMatrix<T>*[total_batch_size];
    DenseMatrix <T> ** c = new DenseMatrix<T>*[total_batch_size];
    for (int i = 0; i < total_batch_size; i++){
      a[i] = new DenseMatrix<T>(m_, k_, parallel_);
      b[i] = new DenseMatrix<T>(k_, n_, parallel_);
      c[i] = new DenseMatrix<T>();
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
      cout << *a[0] << endl;
    double t0 = omp_get_wtime();
    #pragma omp parallel for if (parallel_ == 1)
    for (int i = 0; i < total_batch_size; i++){
      *c[i] = (*a[i]) * (*b[i]);
    }
    double t1 = omp_get_wtime();

    cout << "Time for one matrix multiplication operation with Dense Matrix: " << t1 - t0 << endl;
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
void performSparseMatrixOperations(const int m_, const int n_, const int k_, const int total_batch_size, const int parallel_, const int maxNum){
    SparseMatrix <T> ** a = new SparseMatrix<T>*[total_batch_size]; 
    SparseMatrix <T> ** b = new SparseMatrix<T>*[total_batch_size];
    SparseMatrix <T> ** c = new SparseMatrix<T>*[total_batch_size];
    for (int i = 0; i < total_batch_size; i++){         //initialize every element within the batch
      a[i] = new SparseMatrix<T>(m_, k_, rand() % ((m_*k_)/2), maxNum, parallel_);
      b[i] = new SparseMatrix<T>(k_, n_, rand() % ((k_*n_)/2), maxNum, parallel_);
      c[i] = new SparseMatrix<T>();
    }
    double t0 = omp_get_wtime();
    #pragma omp parallel for if (parallel_ == 1)
    for (int i = 0; i < total_batch_size; i++){     //multiply every matrix within the batch arrays
      *c[i] = (*a[i]) * (*b[i]);
    }
    double t1 = omp_get_wtime();
    cout << "Time for one matrix multiplication operation with Dense Matrix: " << t1 - t0 << endl;
    for (int i = 0; i < total_batch_size; i++) {
      delete a[i];
      delete b[i];
      delete c[i];
    }
    delete[] a;
    delete[] b;
    delete[] c;
}