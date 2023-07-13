#ifndef SPARSEMATRIX_H
#define SPARSEMATRIX_H

// #include <map>
// #include <random>
// #include <time.h>
// #include <utility>
// #include <iostream>
// #include <omp.h>
// #include <algorithm>

using namespace std;

template <typename T>
class SparseMatrix{
    private:
        std::map < pair < int, int > , T > eleLocation_;
        int rows_, cols_, nonZeros_, parallel_; 

    public:
        SparseMatrix(int Row = 0, int Col = 0, int NonZeros = 0, T high = 0, int parallel = 0);

        ~SparseMatrix();

        friend ostream& operator<<(ostream& os, const SparseMatrix<T> & m){
            auto itr = m.eleLocation_.begin();
            bool flag = true;       //corresponds to whether theres still non-zero elements to be outputted
            for (int i = 0; i < m.getRows(); i++){
                for (int j = 0; j < m.getCols(); j++){
                    if (m.getNumNonZeros() > 0 && flag == true && i == itr->first.first && j == itr->first.second){
                        if constexpr(is_same_v<T, MKL_F16>){
                            os << h2f(itr->second) << " ";
                        }
                        else if constexpr(is_same_v<T, boost::multiprecision::float128>){
                            os.setf(std::ios_base::showpoint);
                            os << setprecision(numeric_limits<boost::multiprecision::float128>::max_digits10) << itr->second << " ";
                        }
                        else{
                            os << itr->second << " ";
                        }
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
SparseMatrix<T> :: SparseMatrix(int Row, int Col, int NonZeros, T high, int parallel){
    rows_ = Row;
    cols_ = Col;
    nonZeros_ = NonZeros;
    parallel_ = parallel;
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
    parallel_ = rhs.parallel_;
}

template <typename T>
SparseMatrix<T> :: SparseMatrix(SparseMatrix<T> && rhs){
    rows_ = std::move(rhs.rows_);
    cols_ = std::move(rhs.cols_);
    nonZeros_ = std::move(rhs.nonZeros_);
    eleLocation_ = std::move(rhs.eleLocation_);
    parallel_ = std::move(rhs.parallel_);
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
    std::swap(this->parallel_, copy.parallel_);
    return *this;
}

template <typename T>
SparseMatrix<T>& SparseMatrix<T> :: operator=(SparseMatrix<T>&& rhs){
    std::swap(this->rows_, rhs.rows_);
    std::swap(this->cols_, rhs.cols_);
    std::swap(this->nonZeros_, rhs.nonZeros_);
    std::swap(this->eleLocation_, rhs.eleLocation_);
    std::swap(this->parallel_, rhs.parallel_);
    return *this;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T> :: operator+(const SparseMatrix<T>& rhs){
        SparseMatrix<T> resMatrix(*this);
        resMatrix.rows_ = max(rhs.getRows(), getRows());
        resMatrix.cols_ = max(rhs.getCols(), getCols());
        resMatrix.nonZeros_ = rhs.nonZeros_ + nonZeros_;
        for (auto itr = rhs.eleLocation_.begin(); itr != rhs.eleLocation_.end(); itr++){      //matrix addition operation, if the position already exist, add onto the existing val, otherwise set equal
            if (resMatrix.eleLocation_[itr->first])
            {
                if constexpr(is_same_v<T,MKL_F16>){
                    resMatrix.eleLocation_[itr->first] =  f2h(h2f(resMatrix.eleLocation_[itr->first]) + h2f(itr->second));      //convert from mkl_f16 to float for arithmetic operations, then change back with f2h for storage
                }
                else{
                    resMatrix.eleLocation_[itr->first] += itr->second;
                }
            }
            else{
                if constexpr(is_same_v<T,MKL_F16>){
                    resMatrix.eleLocation_[itr->first] =  f2h(itr->second);
                }
                else{
                    resMatrix.eleLocation_[itr->first] = itr->second;
                }
            }
        }
        return resMatrix;
}

template<typename T>
SparseMatrix<T>&  SparseMatrix<T> :: operator+=(const SparseMatrix<T>& rhs){
    rows_ = max(rhs.getRows(), rows_);
    cols_ = max(rhs.getCols(), cols_);
    nonZeros_ = rhs.nonZeros_ + nonZeros_;
    for (auto itr = rhs.eleLocation_.begin(); itr != rhs.eleLocation_.end(); itr++){            //same thing as above, but change the current matrix
        if (eleLocation_[itr->first]){
            if constexpr(is_same_v<T,MKL_F16>){
                eleLocation_[itr->first] =  f2h(h2f(eleLocation_[itr->first]) + h2f(itr->second));
            }
            else{
                eleLocation_[itr->first] += itr->second;
            }
        }
        else{
            if constexpr(is_same_v<T,MKL_F16>){
                eleLocation_[itr->first] =  f2h(itr->second);
            }
            else{
                eleLocation_[itr->first] = itr->second;
            }
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
                if constexpr(is_same_v<T,MKL_F16>){
                    resMatrix.eleLocation_[itr->first] =  f2h(h2f(resMatrix.eleLocation_[itr->first]) - h2f(itr->second));      //convert from mkl_f16 to float for arithmetic operations, then change back with f2h for storage
                }
                else{
                    resMatrix.eleLocation_[itr->first] -= itr->second;
                }
            }
            else{
                if constexpr(is_same_v<T,MKL_F16>){
                    resMatrix.eleLocation_[itr->first] =  f2h(itr->second);
                }
                else{
                    resMatrix.eleLocation_[itr->first] = itr->second;
                }
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
            if constexpr(is_same_v<T,MKL_F16>){
                eleLocation_[itr->first] =  f2h(h2f(eleLocation_[itr->first]) - h2f(itr->second));
            }
            else{
                eleLocation_[itr->first] -= itr->second;
            }
        }
        else{
            if constexpr(is_same_v<T,MKL_F16>){
                eleLocation_[itr->first] =  f2h(itr->second);
            }
            else{
                eleLocation_[itr->first] = itr->second;
            }
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
    // #pragma omp parallel if (parallel_ == 1)
    for (auto itr = eleLocation_.begin(); itr != eleLocation_.end(); itr++){ 
        for (auto itr2 = rhsT.eleLocation_.begin(); itr2 != rhsT.eleLocation_.end(); itr2++){           //compare the map between two matrices, if theres elements at the same row/col, multiply them and add onto the result matrix row matches
            // #pragma omp task
            if (itr->first.second == itr2->first.second){
                auto pos = std::make_pair(itr->first.first, itr2->first.first);
                if (resMatrix.eleLocation_[pos]){
                    if constexpr(is_same_v<T, MKL_F16>){
                        resMatrix.eleLocation_[pos] = f2h( h2f(resMatrix.eleLocation_[pos]) + (h2f(itr->second) * h2f(itr2->second)));
                    }
                    else{
                        resMatrix.eleLocation_[pos] += (itr->second * itr2->second);
                    }
                }
                else{
                    if constexpr(is_same_v<T, MKL_F16>){
                        resMatrix.eleLocation_[pos] = f2h(h2f(itr->second) * h2f(itr2->second));
                    }
                    else{
                        resMatrix.eleLocation_[pos] = itr->second * itr2->second;
                    }
                }
            }
        }
    }
        resMatrix.nonZeros_ = resMatrix.eleLocation_.size();
    return resMatrix;

}


template <typename T>
SparseMatrix<T>& SparseMatrix<T> :: operator*=(const SparseMatrix<T>& rhs){
    *this = (*this) * rhs;
    return *this;
}

template <typename T>
vector<T> SparseMatrix<T> :: operator*(const vector<T>& rhs){
    vector<T> resVector(rows_, 0);
    if (rhs.size() == cols_){
        int curCol = 0;
        T sum;
        #pragma omp parallel for reduction(+:resVector) if (parallel_ == 1)
        for (auto itr = eleLocation_.begin(); itr != eleLocation_.end(); itr++){            //go through every element of the map and multiply the entries between the matrix and vector where the position match
            if constexpr(is_same_v<T, MKL_F16>){
                resVector[itr->first.first] = f2h(h2f(resVector[itr->first.first]) + h2f(itr->second) * h2f(rhs[itr->first.second]));
            }
            else{
                resVector[itr->first.first] += itr->second * rhs[itr->first.second];
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
        if constexpr(is_same_v<T, MKL_F16>){
            itr->second = f2h((h2f(itr->second) + h2f(rhs)));
        }
        else{
            itr->second += rhs;
        }  
    }
    return resMatrix;
}


template <typename T>
SparseMatrix<T> SparseMatrix<T> :: operator-(const T& rhs){
    SparseMatrix<T> resMatrix(*this);
    for (auto itr = resMatrix.eleLocation_.begin(); itr != resMatrix.eleLocation_.end(); itr++){            //subtraction of rhs to each element
        if constexpr(is_same_v<T, MKL_F16>){
            itr->second = f2h((h2f(itr->second) - h2f(rhs)));
        }
        else{
            itr->second -= rhs;
        }  
    }
    return resMatrix;
}

template <typename T>
SparseMatrix<T> SparseMatrix<T>::operator*(const T& rhs) {
    SparseMatrix<T> resMatrix(*this);
    for (auto itr = resMatrix.eleLocation_.begin(); itr != resMatrix.eleLocation_.end(); itr++){            //multiplication to each element
        if constexpr(is_same_v<T, MKL_F16>){
            itr->second = f2h((h2f(itr->second) * h2f(rhs)));
        }
        else{
            itr->second *= rhs;
        }  
    }
    return resMatrix;
}


template <typename T>
SparseMatrix<T> SparseMatrix<T> :: operator/(const T& rhs){
    SparseMatrix<T> resMatrix(*this);
    for (auto itr = resMatrix.eleLocation_.begin(); itr != resMatrix.eleLocation_.end(); itr++){
        if constexpr(is_same_v<T, MKL_F16>){
            itr->second = f2h((h2f(itr->second) / h2f(rhs)));
        }
        else{
            itr->second /= rhs;
        }    
    }
    return resMatrix;
}
#endif