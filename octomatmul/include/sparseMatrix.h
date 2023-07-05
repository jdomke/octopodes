#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

#include <map>
#include <random>
#include <time.h>
#include <utility>
#include <iostream>
#include <omp.h>
#include <algorithm>
#include "helper.h"
#include <quadmath.h>

using namespace std;

template <typename T>
class SparseMatrix{
    private:
        std::map < pair < int, int > , T > eleLocation_;
        int rows_, cols_, nonZeros_; 

    public:
        SparseMatrix(int Row = 0, int Col = 0, int NonZeros = 0, T high = 0);

        ~SparseMatrix();

        friend ostream& operator<<(ostream& os, const SparseMatrix<T> & m){
            if constexpr(std::is_same_v<T, __float128>){
                
            }
            else{
                auto itr = m.eleLocation_.begin();
                bool flag = true;       //corresponds to whether theres still non-zero elements to be outputted
                for (int i = 0; i < m.getRows(); i++){
                    for (int j = 0; j < m.getCols(); j++){
                        if (m.getNumNonZeros() > 0 && flag == true && i == itr->first.first && j == itr->first.second){
                            os << itr->second << " ";
                            if (itr != m.eleLocation_.end()){
                                itr++;
                            }
                            if (itr == m.eleLocation_.end()){
                                flag = false;
                            }
                        }
                        else{
                            os << "0 ";
                        }
                    }
                    os << endl;
                }
            }
            // for (auto itr = m.eleLocation_.begin(); itr != m.eleLocation_.end(); itr++){
            //     cout << itr->first.first << " " << itr->first.second << " " << itr->second << endl;
            // }
            return os;            
        } 

        SparseMatrix(const SparseMatrix<T>& rhs);

        SparseMatrix(SparseMatrix<T> && rhs);

        void insert(int row, int col, T val);

        SparseMatrix<T>& operator=(const SparseMatrix<T>& rhs);

        SparseMatrix<T>& operator=(SparseMatrix<T>&& rhs);

        SparseMatrix<T> operator+(const SparseMatrix<T>& rhs);

        SparseMatrix<T>& operator+=(const SparseMatrix<T>& rhs);

        SparseMatrix<T> operator-(const SparseMatrix<T> & rhs);

        SparseMatrix<T>& operator-=(const SparseMatrix<T> & rhs);

        SparseMatrix<T> operator*(const SparseMatrix<T>& rhs);

        SparseMatrix<T>& operator*=(const SparseMatrix<T>& rhs);

        SparseMatrix<T> Transpose();

        vector<T> operator*(const vector<T>& rhs);

        SparseMatrix<T> operator+(const T& rhs);

        SparseMatrix<T> operator-(const T& rhs);

        SparseMatrix<T> operator*(const T& rhs);

        SparseMatrix<T> operator/(const T& rhs);


        int getRows() const;

        int getCols() const;

        int getNumNonZeros() const;

};

template<typename T>
SparseMatrix<T> :: SparseMatrix(int Row, int Col, int NonZeros, T high){
    rows_ = Row;
    cols_ = Col;
    nonZeros_ = NonZeros;
    map<pair<int,int>, int> posDupe;
    srand(10);
    if (high > 0){
        for (int i = 0; i < nonZeros_; i++){
            int row = rand() % Row;         //obtain the number of nonZeros and generate that many random elements for the matrix
            int col = rand() % Col;
            T val = getRandomValue<T>(0, high);
            std::pair <int, int> pos = std::make_pair(row,col);    
            while (posDupe.find(pos)!= posDupe.end()){          //check if the generated row/column pair is duplicated, if so, generate new until no duplicate
                row = rand() % Row;
                col = rand() % Col;
                pos = std::make_pair(row, col);
                if (posDupe.size() == Row * Col){
                    exit(1);
                }
            }
            posDupe[pos] = 1;
            eleLocation_[pos] = val;
        }
    }
    // for (auto itr = eleLocation_.begin(); itr != eleLocation_.end(); itr++){
    //     cout << itr->first.first << " " << itr->first.second << " " << itr->second << endl;
    // }
}

template <typename T>
SparseMatrix<T> :: SparseMatrix(const SparseMatrix<T> & rhs){
    rows_ = rhs.rows_;
    cols_ = rhs.cols_;
    nonZeros_ = rhs.nonZeros_;
    eleLocation_ = rhs.eleLocation_;
}

template <typename T>
SparseMatrix<T> :: SparseMatrix(SparseMatrix<T> && rhs){
    rows_ = std::move(rhs.rows_);
    cols_ = std::move(rhs.cols_);
    nonZeros_ = std::move(rhs.nonZeros_);
    eleLocation_ = std::move(rhs.eleLocation_);
}
template <typename T>
SparseMatrix<T> :: ~SparseMatrix() = default;

template <typename T>
void SparseMatrix<T> :: insert(int row, int col, T val){
    nonZeros_++;
    eleLocation_[std::make_pair(row, col)] = val;
}
template <typename T>
SparseMatrix<T>& SparseMatrix<T> :: operator=(const SparseMatrix<T>& rhs){
    SparseMatrix<T> copy(rhs);
    std::swap(this->rows_, copy.rows_);
    std::swap(this->cols_, copy.cols_);
    std::swap(this->nonZeros_, copy.nonZeros_);
    std::swap(this->eleLocation_, copy.eleLocation_);
    return *this;
}

template <typename T>
SparseMatrix<T>& SparseMatrix<T> :: operator=(SparseMatrix<T>&& rhs){
    std::swap(this->rows_, rhs.rows_);
    std::swap(this->cols_, rhs.cols_);
    std::swap(this->nonZeros_, rhs.nonZeros_);
    std::swap(this->eleLocation_, rhs.eleLocation_);
    return *this;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T> :: operator+(const SparseMatrix<T>& rhs){
        SparseMatrix<T> resMatrix(*this);
        resMatrix.rows_ = max(rhs.getRows(), getRows());
        resMatrix.cols_ = max(rhs.getCols(), getCols());
        resMatrix.nonZeros_ = rhs.nonZeros_ + nonZeros_;
        for (auto itr = rhs.eleLocation_.begin(); itr != rhs.eleLocation_.end(); itr++){
            if (resMatrix.eleLocation_[itr->first])
            {
                resMatrix.eleLocation_[itr->first] += itr->second;
            }
            else{
                resMatrix.eleLocation_[itr->first] = itr->second;
            }
        }
        return resMatrix;
}

template<typename T>
SparseMatrix<T>&  SparseMatrix<T> :: operator+=(const SparseMatrix<T>& rhs){
    rows_ = max(rhs.getRows(), rows_);
    cols_ = max(rhs.getCols(), cols_);
    nonZeros_ = rhs.nonZeros_ + nonZeros_;
    for (auto itr = rhs.eleLocation_.begin(); itr != rhs.eleLocation_.end(); itr++){
        if (eleLocation_[itr->first]){
            eleLocation_[itr->first] += itr->second;
        }
        else{
            eleLocation_[itr->first] = itr->second;
        }
    }
    return *this;
}
template <typename T>
SparseMatrix<T> SparseMatrix<T> :: operator-(const SparseMatrix<T> & rhs){
        SparseMatrix<T> resMatrix(*this);
        resMatrix.rows_ = max(rhs.getRows(), getRows());
        resMatrix.cols_ = max(rhs.getCols(), getCols());
        resMatrix.nonZeros_ = rhs.nonZeros_ + nonZeros_;
        for (auto itr = rhs.eleLocation_.begin(); itr != rhs.eleLocation_.end(); itr++){
            if (resMatrix.eleLocation_[itr->first])
            {
                resMatrix.eleLocation_[itr->first] -= itr->second;
            }
            else{
                resMatrix.eleLocation_[itr->first] = -itr->second;
            }
        }
        return resMatrix;
}

template <typename T>
SparseMatrix<T>& SparseMatrix<T> :: operator-=(const SparseMatrix<T> & rhs){
    rows_ = max(rhs.getRows(), rows_);
    cols_ = max(rhs.getCols(), cols_);
    nonZeros_ = rhs.nonZeros_ + nonZeros_;
    for (auto itr = rhs.eleLocation_.begin(); itr != rhs.eleLocation_.end(); itr++){
        if (eleLocation_[itr->first]){
            eleLocation_[itr->first] -= itr->second;
        }
        else{
            eleLocation_[itr->first] = -itr->second;
        }
    }
    return *this;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T> :: Transpose(){
    SparseMatrix<T> resMatrix(cols_, rows_, nonZeros_);
    for (auto itr = eleLocation_.begin(); itr != eleLocation_.end(); itr++)
    {
        pair<int, int> pos = make_pair(itr->first.second, itr->first.first);
        resMatrix.eleLocation_[pos] = itr->second;
        // cout << pos.first << " " << pos.second << " " << resMatrix.eleLocation_[pos] << endl;
    }
    return resMatrix;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T> :: operator*(const SparseMatrix<T>& rhs){
    if (cols_ != rhs.rows_ ){
        cout << "unmatching size between two matrices.\n";
        exit(1);
    }
    auto rhsT(rhs);
    rhsT = rhsT.Transpose();
    SparseMatrix<T> resMatrix(getRows(),rhsT.getRows(), 0, 0);
    #pragma omp parallel
    #pragma omp single
    for (auto itr = eleLocation_.begin(); itr != eleLocation_.end(); itr++){ 
        for (auto itr2 = rhsT.eleLocation_.begin(); itr2 != rhsT.eleLocation_.end(); itr2++){
            #pragma omp task
            if (itr->first.second == itr2->first.second){
                auto pos = std::make_pair(itr->first.first, itr2->first.first);
                if (resMatrix.eleLocation_[pos]){
                    resMatrix.eleLocation_[pos] += (itr->second * itr2->second);
                }
                else{
                    resMatrix.eleLocation_[pos] = itr->second * itr2->second;
                }
            }
        }
    }
    resMatrix.nonZeros_ = resMatrix.eleLocation_.size();
    return resMatrix;

}
// template <typename T>
// SparseMatrix<T> SparseMatrix<T>::operator*(const SparseMatrix<T>& rhs) {
//     if (cols_ != rhs.rows_) {
//         cout << "Unmatching size between two matrices.\n";
//         exit(1);
//     }

//     auto rhsT(rhs);
//     rhsT = rhsT.Transpose();
//     SparseMatrix<T> resMatrix(getRows(), rhsT.getRows(), 0, 0);

//     #pragma omp double
//     for (auto itr = eleLocation_.begin(); itr != eleLocation_.end(); itr++) {
//         for (auto itr2 = rhsT.eleLocation_.begin(); itr2 != rhsT.eleLocation_.end(); itr2++) {
//             #pragma omp task
//             if (itr->first.second == itr2->first.second) {
//                 auto pos = std::make_pair(itr->first.first, itr2->first.first);
//                 if (resMatrix.eleLocation_[pos]) {
//                     resMatrix.eleLocation_[pos] += (itr->second * itr2->second);
//                 } else {
//                     resMatrix.eleLocation_[pos] = itr->second * itr2->second;
//                 }
//             }
//         }
//     }

//     resMatrix.nonZeros_ = resMatrix.eleLocation_.size();
//     return resMatrix;
// }


template <typename T>
SparseMatrix<T>& SparseMatrix<T> :: operator*=(const SparseMatrix<T>& rhs){
    *this = (*this) * rhs;
    return *this;
}

template <typename T>
vector<T> SparseMatrix<T> :: operator*(const vector<T>& rhs){
    vector<T> resVector(rows_, 0);
    if (rhs.size() == cols_){
        int j = 0;
        for (int i = 0; i < rows_; i++){
            j = 0;
            T sum = 0;
            for (auto itr = eleLocation_.begin(); itr != eleLocation_.end(); itr++){
                if (itr->first.first == i){
                    sum += itr->second * rhs[j];
                    j++;
                }
                if (itr->first.first > i){
                    break;
                }
            }
            if (sum > 0){
                resVector[i] = sum;
            }
        }
    }
    else{
        cout << "operation failed due to unmatched size\n";
    }
    return resVector;
}
template <typename T>
int SparseMatrix<T> :: getRows() const{
    return rows_;
}

template <typename T>
int SparseMatrix<T> :: getNumNonZeros() const{
    return nonZeros_;
}

template <typename T>
int SparseMatrix<T> :: getCols() const{
    return cols_;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T> :: operator+(const T& rhs)
{
    SparseMatrix<T> resMatrix(*this);
    for (auto itr = resMatrix.eleLocation_.begin(); itr != resMatrix.eleLocation_.end(); itr++){
        itr->second += rhs;  
    }
    return resMatrix;
}


template <typename T>
SparseMatrix<T> SparseMatrix<T> :: operator-(const T& rhs){
    SparseMatrix<T> resMatrix(*this);
    for (auto itr = resMatrix.eleLocation_.begin(); itr != resMatrix.eleLocation_.end(); itr++){
        itr->second -= rhs;  
    }
    return resMatrix;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::operator*(const T& rhs) {
    SparseMatrix<T> resMatrix(*this);

    std::vector<std::pair<std::pair<int, int>, T>> elements(resMatrix.eleLocation_.begin(), resMatrix.eleLocation_.end());

    for (size_t i = 0; i < elements.size(); ++i) {
        elements[i].second *= rhs;
    }

    resMatrix.eleLocation_.clear();
    for (const auto& element : elements) {
        resMatrix.eleLocation_.insert(element);
    }

    return resMatrix;
}


template <typename T>
SparseMatrix<T> SparseMatrix<T> :: operator/(const T& rhs){
    SparseMatrix<T> resMatrix(*this);
    for (auto itr = resMatrix.eleLocation_.begin(); itr != resMatrix.eleLocation_.end(); itr++){
        itr->second /= rhs;  
    }
    return resMatrix;
}
#endif