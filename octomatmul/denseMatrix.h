#ifndef DENSEMATRIX_H
#define DENSEMATRIX_H

#include <vector>
#include <omp.h>
#include <iostream>

using namespace std;


template <typename T>
class DenseMatrix{

        private: 
            vector< vector< T > >  matrix_;
            int rows_;
            int cols_;

        public:
            DenseMatrix(int rows = 0, int cols = 0, const T& iniVal = 0);

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

            vector<T> operator*(const vector<T>& rhs);

            friend std::ostream& operator<<(std::ostream& os, const DenseMatrix<T>& M){
                for (int i = 0; i < M.getRows(); ++i) {
                    for (int j = 0; j < M.getCols(); ++j) {
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
DenseMatrix<T>::DenseMatrix(int rows, int cols, const T& iniVal){                     //resize DenseMatrix to [ROW][COL] of initial value, defaulted as 0
    matrix_.resize(rows, vector<T>(cols, iniVal));
    rows_ = rows;
    cols_ = cols;
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

}

template <typename T>
DenseMatrix<T>& DenseMatrix<T>::operator=(const DenseMatrix<T>& rhs) {
    DenseMatrix<T> copy(rhs);  // Create a copy of the rhs DenseMatrix
    this->matrix_.resize(rhs.rows_, vector<T>(rhs.cols_));
    std::swap(this->matrix_, copy.matrix_);
    std::swap(this->rows_, copy.rows_);
    std::swap(this->cols_, copy.cols_);
    return *this;
}

template <typename T>

DenseMatrix<T>& DenseMatrix<T>::operator=(DenseMatrix<T>&& rhs) {
    this->matrix_.resize(rhs.rows_, vector<T>(rhs.cols_));
    std::swap(this->matrix_, rhs.matrix_);  // Swap the contents of the current DenseMatrix and the rhs DenseMatrix
    std::swap(this->rows_, rhs.rows_);
    std::swap(this->cols_,  rhs.cols_);
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
                matrix_[i][j] += rhs.matrix_[i][j];
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
                resMatrix.matrix_[i][j] = matrix_[i][j] - rhs.matrix_[i][j];
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
                    matrix_[i][j] -= rhs.matrix_[i][j];
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
    #pragma omp declare reduction(DenseMatrixSum: DenseMatrix<T>: \
    omp_out += omp_in) \
    initializer(omp_priv = DenseMatrix<T>())
    
    DenseMatrix<T> resMatrix(rows_, rhs.cols_); 
    if (cols_ == rhs.rows_){
        #pragma omp parallel for collapse(3) reduction(DenseMatrixSum:resMatrix)
        for (int i = 0; i < rows_; i++){
            for (int j = 0; j < rhs.cols_; j++){
                for (int k = 0; k < cols_; k++){
                    resMatrix.matrix_[i][j] += matrix_[i][k] * rhs.matrix_[k][j];
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
    #pragma omp for collapse(2)
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
    for (int i = 0; i < rows_; i++){
        for (int j = 0; j < cols_; j++){
            resMatrix.matrix_[i][j] = matrix_[i][j] + rhs;
        }
    }
    return resMatrix;
}
template <typename T>
DenseMatrix<T> DenseMatrix<T> :: operator-(const T& rhs){
    DenseMatrix<T> resMatrix(rows_, cols_);
    #pragma omp for collapse(2)
    for (int i = 0; i < rows_; i++){
        for (int j = 0; j < cols_; j++){
            resMatrix.matrix_[i][j] = matrix_[i][j] - rhs;
        }
    }
    return resMatrix;
}

template <typename T>
DenseMatrix<T> DenseMatrix<T> :: operator*(const T& rhs){
    DenseMatrix<T> resMatrix(rows_, cols_);
    #pragma omp for collapse(2)
    for (int i = 0; i < rows_; i++){
        for (int j = 0; j < cols_; j++){
            resMatrix.matrix_[i][j] = matrix_[i][j] * rhs;
        }
    }
    return resMatrix;
}

template <typename T>
DenseMatrix<T> DenseMatrix<T> :: operator/(const T& rhs){
    DenseMatrix<T> resMatrix(rows_, cols_);
    #pragma omp for collapse(2)
    for (int i = 0; i < rows_; i++){
        for (int j = 0; j < cols_; j++){
            resMatrix.matrix_[i][j] = matrix_[i][j] / rhs;
        }
    }
    return resMatrix;
}
template <typename T>
vector<T> DenseMatrix<T> :: operator*(const vector<T>& rhs){
    vector<T> resVector(rows_, 0);
    if (rhs.size() == cols_){
        #pragma omp for collapse(2)
        for (int i = 0; i < rows_; i++){
            for (int j = 0; j < cols_; j++){
                resVector[i] += matrix_[i][j]* rhs[j];
            }
        }
    }   
    else{
        cout << "operation failed due to unmatched size\n";
    }
    return resVector;
}


#endif
