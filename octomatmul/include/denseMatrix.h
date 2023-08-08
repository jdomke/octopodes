#ifndef DENSEMATRIX_H
#define DENSEMATRIX_H

using namespace std;


template <typename T>
class DenseMatrix{
        private: 
            vector< vector< T > >  matrix_;
            int rows_;
            int cols_;
            int parallel_;
        public:
            DenseMatrix(int rows = 0, int cols = 0, int parallel = 0, const T& iniVal = 0);

            DenseMatrix(const DenseMatrix<T> & rhs);

            DenseMatrix(DenseMatrix<T> && rhs);
            
            ~DenseMatrix();

            DenseMatrix<T>& operator=(const DenseMatrix<T>& rhs);

            DenseMatrix<T>& operator=(DenseMatrix<T>&& rhs);

            DenseMatrix<T> operator+(const DenseMatrix<T>& rhs);

            DenseMatrix<T>& operator+=(const DenseMatrix<T>& rhs);

            DenseMatrix<T> operator-(const DenseMatrix<T>& rhs);

            DenseMatrix<T>& operator-=(const DenseMatrix<T>& rhs);

            DenseMatrix<T> operator*(const DenseMatrix<T>& rhs);

            DenseMatrix<T>& operator*=(const DenseMatrix<T>& rhs);
            
            DenseMatrix<T> Transpose();

            DenseMatrix<T> operator+(const T& rhs);

            DenseMatrix<T> operator-(const T& rhs);

            DenseMatrix<T> operator*(const T& rhs);

            DenseMatrix<T> operator/(const T& rhs);

            void insert(int col, int row, T val);

            vector<T> operator*(const vector<T>& rhs);

            friend std::ostream& operator<<(std::ostream& os, const DenseMatrix<T>& M){
                for (int i = 0; i < M.getRows(); ++i) {
                    for (int j = 0; j < M.getCols(); ++j) {
                        if constexpr(is_same_v<T, MKL_F16>){
                            os << h2f(M.matrix_[i][j]) << ' ';
                        }
                        else if constexpr(is_same_v<T, boost::multiprecision::float128>){
                            os.setf(std::ios_base::showpoint);
                            os << setprecision(numeric_limits<boost::multiprecision::float128>::max_digits10) << M.matrix_[i][j] <<  ' ';
                        }
                        else
                           os << M.matrix_[i][j] << ' ';   
                    }
                    os << '\n';
                }     
                return os;
            }
                
            T& operator()(int row, int col);

            const T& operator()(int row, int col) const;

            int getRows() const;

            int getCols() const;

};

template <typename T>
DenseMatrix<T>::DenseMatrix(int rows, int cols, int parallel, const T& iniVal){                     //resize DenseMatrix to [ROW][COL] of initial value, defaulted as 0
    matrix_.resize(rows, vector<T>(cols, iniVal));
    rows_ = rows;
    cols_ = cols;
    parallel_ = parallel;
}

template <typename T>
DenseMatrix<T>::DenseMatrix(const DenseMatrix<T>& rhs){
    matrix_.resize(rhs.rows_);
    for (int i = 0; i < rhs.rows_; i++) {
        matrix_[i].resize(rhs.cols_);
        for (int j = 0; j < rhs.cols_; j++) {
            matrix_[i][j] = rhs.matrix_[i][j];
        }
    }
    rows_ = rhs.rows_;
    cols_ = rhs.cols_;
    parallel_ = rhs.parallel_;

}

template <typename T>
DenseMatrix<T>& DenseMatrix<T>::operator=(const DenseMatrix<T>& rhs) {
    DenseMatrix<T> copy(rhs);  // Create a copy of the rhs DenseMatrix
    this->matrix_.resize(rhs.rows_, vector<T>(rhs.cols_));
    std::swap(this->matrix_, copy.matrix_);
    std::swap(this->rows_, copy.rows_);
    std::swap(this->cols_, copy.cols_);
    std::swap(this->parallel_, copy.parallel_);
    return *this;
}

template <typename T>

DenseMatrix<T>& DenseMatrix<T>::operator=(DenseMatrix<T>&& rhs) {
    this->matrix_.resize(rhs.rows_, vector<T>(rhs.cols_));
    std::swap(this->matrix_, rhs.matrix_);  // Swap the contents of the current DenseMatrix and the rhs DenseMatrix
    std::swap(this->rows_, rhs.rows_);
    std::swap(this->cols_,  rhs.cols_);
    std::swap(this->parallel_, rhs.parallel_);
    rhs.rows_ = 0;
    rhs.cols_ = 0;
    return *this;
}

template <typename T>
DenseMatrix<T>::DenseMatrix(DenseMatrix<T> && rhs){
    matrix_.resize(rhs.rows_);            //resize original DenseMatrix to fit the new DenseMatrix and move oveer everything
    for (int i = 0; i < rhs.rows_; i++) {
        matrix_[i].resize(rhs.cols_);
        for (int j = 0; j < rhs.cols_; j++) {
            matrix_[i][j] = std::move(rhs.matrix_[i][j]);
        }
    }
    rows_ = rhs.rows_;
    cols_ = rhs.cols_;
    parallel_ = rhs.parallel_;
    rhs.rows_ = 0;
    rhs.cols_ = 0;

}

template <typename T>
DenseMatrix<T>::~DenseMatrix() = default;

template <typename T>
int DenseMatrix<T> :: getRows() const{
    return rows_;
}

template <typename T>
int DenseMatrix<T> :: getCols() const{
    return cols_;   
}

template <typename T>
DenseMatrix<T> DenseMatrix<T> :: operator+(const DenseMatrix<T>& rhs){
    DenseMatrix<T> resMatrix(rows_, cols_);

    if (rows_ == rhs.rows_ && cols_ == rhs.cols_){
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < rows_; i++){
            for (int j = 0; j < cols_; j++){
                if constexpr(is_same_v<T,MKL_F16>){
                    resMatrix.matrix_[i][j] = f2h(h2f(matrix_[i][j]) + h2f(rhs.matrix_[i][j]));
                }
                else
                    resMatrix.matrix_[i][j] = matrix_[i][j] + rhs.matrix_[i][j];
            }
        }
    }
    else{
        cout << "Unable to proceed due to unmatching No. of Rows and Columns.\n";
    }
    return resMatrix;
}
template <typename T>
DenseMatrix<T> &DenseMatrix<T> :: operator+=(const DenseMatrix<T>& rhs){
    if (rows_ == rhs.rows_ && cols_ == rhs.cols_){            //adding each entry of the two matrices together 
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < rows_; i++){
            for (int j = 0; j < cols_; j++){
                if constexpr(is_same_v<T,MKL_F16>){
                    matrix_[i][j] = f2h(h2f(matrix_[i][j]) + h2f(rhs.matrix_[i][j]));
                }
                else{
                    matrix_[i][j] += rhs.matrix_[i][j];
                }
            }
        }
    }
    else{
        cout << "Unable to proceed due to unmatching No. of Rows and Columns.\n";
    }
    return *this;
}

template <typename T>
DenseMatrix<T> DenseMatrix<T> :: operator-(const DenseMatrix<T>& rhs){
    DenseMatrix<T> resMatrix(rows_, cols_);
    if (rows_ == rhs.rows_ && cols_ == rhs.cols_){            //distribute the subtraction operation to multiple threads
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < rows_; i++){
            for (int j = 0; j < cols_; j++){
                if constexpr(is_same_v<T,MKL_F16>){
                    resMatrix.matrix_[i][j] = f2h(h2f(matrix_[i][j]) - h2f(rhs.matrix_[i][j]));
                }
                else{
                    resMatrix.matrix_[i][j] = matrix_[i][j] - rhs.matrix_[i][j];
                }
            }
        }
    }
    else{
        cout << "Unable to proceed due to unmatching No. of Rows and Columns.\n";
    }
    return resMatrix;
}

template <typename T>
DenseMatrix<T> &DenseMatrix<T> :: operator-=(const DenseMatrix<T>& rhs){
    if (rows_ == rhs.rows_ && cols_ == rhs.cols_){                //subtract DenseMatrix with omp
            #pragma omp parallel for collapse(2) 
            for (int i = 0; i < rows_; i++){
                for (int j = 0; j < cols_; j++){
                    if constexpr(is_same_v<T,MKL_F16>){
                        matrix_[i][j] = f2h(h2f(matrix_[i][j]) - h2f(rhs.matrix_[i][j]));
                    }
                    else{
                        matrix_[i][j] -= rhs.matrix_[i][j];
                    }
                }
            }
    }
    else{
        cout << "Unable to proceed due to unmatching No. of Rows and Columns.\n";
    }
    return *this;
}

template <typename T>
DenseMatrix<T> DenseMatrix<T> :: operator*(const DenseMatrix<T>& rhs){

    DenseMatrix<T> resMatrix(rows_, rhs.cols_, 0); 
    if (cols_ == rhs.rows_){
            DenseMatrix<T> tempMatrix(rows_, rhs.cols_);
            #pragma omp parallel for collapse(3) if (parallel_ == 1)                //standard 3 nested loop algorithm for matrix multiplication. if type is mkl_f16, convert to float and perform operation, then convert back
            for (int i = 0; i < rows_; i++) {   
                for (int j = 0; j < rhs.cols_; j++) {
                    for (int k = 0; k < cols_; k++) {
                        if constexpr(is_same_v<T,MKL_F16>){
                            tempMatrix.matrix_[i][j] = f2h(h2f(tempMatrix.matrix_[i][j]) + h2f(matrix_[i][k]) * h2f(rhs.matrix_[k][j]));
                        }
                        else{
                            tempMatrix.matrix_[i][j] += matrix_[i][k] * rhs.matrix_[k][j];
                        }
                    }
                }
            }
            
            #pragma omp critical                 //append tempMatrix elements to resMatrix
            {
                resMatrix += tempMatrix;
            }
    }
    else{
        cout << "Unable to proceed due to unmatching No. of Rows and Columns.\n";
    }
    return resMatrix;
}
template <typename T>

DenseMatrix<T>& DenseMatrix<T> ::  operator*=(const DenseMatrix<T>& rhs){
    *this = (*this) * rhs;
    return *this;

    
}
template <typename T>
T& DenseMatrix<T> :: operator()(int row, int col){
    if (row < rows_ && col < cols_){
        return matrix_[row][col];
    }
    else{
        cout << "Index out of bounds\n";
        exit(1);
    }

}
template <typename T>
const T& DenseMatrix<T> :: operator()(int row, int col) const{
    if (row < rows_ && col < cols_){
        return matrix_[row][col];
    }
    else{
        cout << "Index out of bounds\n";
        exit(1);
    }
}
template <typename T>
DenseMatrix<T> DenseMatrix <T> :: Transpose(){
    DenseMatrix<T> resMatrix(cols_, rows_);
    #pragma omp for collapse(2)             //swap row and col on every value
    for (int i = 0; i < rows_; i++){
        for (int j = 0; j < cols_; j++){
            resMatrix.matrix_[j][i] = matrix_[i][j];
        }
    }
    return resMatrix;
}
template <typename T>
DenseMatrix<T> DenseMatrix<T> :: operator+(const T& rhs){
    DenseMatrix<T> resMatrix(rows_, cols_);
    #pragma omp for collapse(2)
    for (int i = 0; i < rows_; i++){            //go through every entry and add it to rhs val
        for (int j = 0; j < cols_; j++){
            if constexpr(is_same_v<T,MKL_F16>){
                resMatrix.matrix_[i][j] = f2h(h2f(matrix_[i][j])  + h2f(rhs));
            }
            else{
                resMatrix.matrix_[i][j] = matrix_[i][j] + rhs;
            }
        }
    }
    return resMatrix;
}
template <typename T>
DenseMatrix<T> DenseMatrix<T> :: operator-(const T& rhs){
    DenseMatrix<T> resMatrix(rows_, cols_);         //same thing but subtract
    #pragma omp for collapse(2)
    for (int i = 0; i < rows_; i++){
        for (int j = 0; j < cols_; j++){
            if constexpr(is_same_v<T,MKL_F16>){
                resMatrix.matrix_[i][j] = f2h(h2f(matrix_[i][j]) - h2f(rhs));;
            }
            else{
                resMatrix.matrix_[i][j] = matrix_[i][j] - rhs;
            }
        }
    }
    return resMatrix;
}

template <typename T>
DenseMatrix<T> DenseMatrix<T> :: operator*(const T& rhs){
    DenseMatrix<T> resMatrix(rows_, cols_);         //same thing but multiply
    #pragma omp for collapse(2)
    for (int i = 0; i < rows_; i++){
        for (int j = 0; j < cols_; j++){
            if constexpr(is_same_v<T,MKL_F16>){
                resMatrix.matrix_[i][j] = f2h(h2f(matrix_[i][j]) * h2f(rhs));;
            }
            else{
                resMatrix.matrix_[i][j] = matrix_[i][j] * rhs;
            }
        }
    }
    return resMatrix;
}

template <typename T>
DenseMatrix<T> DenseMatrix<T> :: operator/(const T& rhs){
    DenseMatrix<T> resMatrix(rows_, cols_);         //same thing but for division
    #pragma omp for collapse(2)
    for (int i = 0; i < rows_; i++){
        for (int j = 0; j < cols_; j++){
            if constexpr(is_same_v<T,MKL_F16>){
                resMatrix.matrix_[i][j] = f2h(h2f(matrix_[i][j]) / h2f(rhs));;
            }
            else{
                resMatrix.matrix_[i][j] = matrix_[i][j] / rhs;
            }
        }
    }
    return resMatrix;
}

template <typename T>
vector<T> DenseMatrix<T>::operator*(const vector<T>& rhs) {
    vector<T> resVector(rows_, 0);

    if (rhs.size() == cols_) {
        #pragma omp parallel for if (parallel_ == 1)
        for (int i = 0; i < rows_; i++) {
            T localResult = 0; // Use a local variable to store the result to prevent incorrect result from parallelization
            for (int j = 0; j < cols_; j++) {
                if constexpr (is_same_v<T, MKL_F16>) {
                    localResult = f2h(h2f(localResult) + h2f(matrix_[i][j]) * h2f(rhs[j]));
                } else {
                    localResult += matrix_[i][j] * rhs[j];
                }
            }
            resVector[i] = localResult; // Update the result vector after the inner loop
        }
    } else {
        cout << "Operation failed due to unmatched size\n";
    }

    return resVector;
}
template <typename T>
void DenseMatrix<T> :: insert(int row, int col, T val){
    if (row < rows_ && col < cols_){
        matrix_[row][col] = val;
    }
}

#endif
