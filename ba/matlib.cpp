// Matrix class
template <class T>
class Mat{
  
  private:
    unsigned rows_, cols_;
    int rowstride_;
    T * p_;

  public:
    // Constructor  which is automatically called when the object is created. 
    // Must have same name as the class type.
    Mat(T * p, unsigned rows, unsigned cols, int rowstride = 0) 
    {
      p_ = p;
      rows_ = rows;
      cols_ = cols;
      rowstride_ = rowstride; // How many entries in memory each row is separated by
    }

    // Get functions
    unsigned getnCols() const
    {
      return cols_;
    }

    unsigned getnRows() const
    {
      return rows_;
    }

    //Overload () operator to retrieve the element of the matrix at address p in row r and column c
    T& operator()(unsigned r, unsigned c)
    {
      return p_[r * (cols_ + rowstride_) + c];
    }

    const T& operator()(unsigned r, unsigned c) const
    {
      return  p_[r * (cols_ + rowstride_) + c];
    }

};

// Basic operations

template <class T, class U, class V>
void matMul(const Mat<T>& mat1, const Mat<U>& mat2, Mat<V>& product, bool transpose1 = false, bool transpose2 = false)
{
  if ((transpose1 == false) && (transpose2 == false)) {
    for (unsigned i = 0; i < mat1.getnRows(); ++i) {
      for (unsigned j = 0; j < mat2.getnCols(); ++j) {
        for (unsigned k = 0; k < mat1.getnCols(); ++k) {
            product(i, j) += mat1(i,k) * mat2(k,j);
        }
      }
    }
  }

  else if ((transpose1 == true) && (transpose2 == false)) {
    for (unsigned i = 0; i < mat1.getnCols(); ++i) {
      for (unsigned j = 0; j < mat2.getnCols(); ++j) {
        for (unsigned k = 0; k < mat1.getnRows(); ++k) {
            product(i, j) += mat1(k,i) * mat2(k,j);
        }
      }
    }
  }

  else if ((transpose1 == false) && (transpose2 == true)) {
    for (unsigned i = 0; i < mat1.getnRows(); ++i) {
      for (unsigned j = 0; j < mat2.getnRows(); ++j) {
        for (unsigned k = 0; k < mat1.getnCols(); ++k) {
            product(i, j) += mat1(i,k) * mat2(j,k);
        }
      }
    }
  }        

  else if ((transpose1 == true) && (transpose2 == true)) {
    for (unsigned i = 0; i < mat1.getnCols(); ++i) {
      for (unsigned j = 0; j < mat2.getnRows(); ++j) {
        for (unsigned k = 0; k < mat1.getnRows(); ++k) {
            product(i, j) += mat1(k,i) * mat2(j,k);
        }
      }
    }
  } 
}

template <class T, class U, class V>
void matSum(const Mat<T>& mat1, const Mat<U>& mat2, Mat<V>& sum, bool subtract = false)
{
  // // you could use a constructor of Mat<float> that alreay allocates the memory for you
  // float* array = new float[this->rows * otherMat.getnCols()]; // allocate mem for result
  // Mat<float> res(array, this->rows, otherMat.getnCols()); // create a matrix view
  
  // Matrix dimensions must be equal.
  if (( mat1.getnRows() == mat2.getnRows() ) && ( mat1.getnCols() == mat2.getnCols() )) {

    for (unsigned i = 0; i < mat1.getnRows(); ++i) {
      for (unsigned j = 0; j < mat1.getnCols(); ++j) {
        if (subtract == false){
          sum(i,j) = mat1(i,j) + mat2(i,j);
        }
        else {
          sum(i,j) = mat1(i,j) - mat2(i,j);
        }
      }
    }
  }  
}


template <class T, class U, class V>
void outerProduct(const Mat<T>& u, const Mat<U>& v, Mat<V>& outer)
{
  // u and v must be column vectors
  if (( u.getnCols() == 1 ) && ( v.getnCols() == 1 )) {
    for (unsigned i = 0; i < u.getnRows(); ++i) {
      for (unsigned j = 0; j < v.getnRows(); ++j) {
        outer(i,j) = u(i,0) * v(j,0);
      }
    }
  }  
}

template <class T, class U>
float dotProduct(const Mat<T>& u, const Mat<U>& v)
{
  // u and v must be column vectors with equal length
  float uv = 0;
  if (( u.getnCols() == 1 ) && ( v.getnCols() == 1 ) && (u.getnRows() == v.getnRows())) {
    for (unsigned i = 0; i < u.getnRows(); ++i) {
      uv += u(i,0) * v(i,0);
    }
  }  
  return uv;
}

// Matrix Inversion

template <class T, class U>
void inv3x3(const Mat<T>& M, Mat<U>& inv)
{  
    // computes the inverse of a matrix m
    float det = M(0, 0) * (M(1, 1) * M(2, 2) - M(2, 1) * M(1, 2)) -
                 M(0, 1) * (M(1, 0) * M(2, 2) - M(1, 2) * M(2, 0)) +
                 M(0, 2) * (M(1, 0) * M(2, 1) - M(1, 1) * M(2, 0));

    inv(0, 0) = (M(1, 1) * M(2, 2) - M(2, 1) * M(1, 2)) / det;
    inv(0, 1) = (M(0, 2) * M(2, 1) - M(0, 1) * M(2, 2)) / det;
    inv(0, 2) = (M(0, 1) * M(1, 2) - M(0, 2) * M(1, 1)) / det;
    inv(1, 0) = (M(1, 2) * M(2, 0) - M(1, 0) * M(2, 2)) / det;
    inv(1, 1) = (M(0, 0) * M(2, 2) - M(0, 2) * M(2, 0)) / det;
    inv(1, 2) = (M(1, 0) * M(0, 2) - M(0, 0) * M(1, 2)) / det;
    inv(2, 0) = (M(1, 0) * M(2, 1) - M(2, 0) * M(1, 1)) / det;
    inv(2, 1) = (M(2, 0) * M(0, 1) - M(0, 0) * M(2, 1)) / det;
    inv(2, 2) = (M(0, 0) * M(1, 1) - M(1, 0) * M(0, 1)) / det;

}

template <class T, class U>
void inv_uppertriang(const Mat<T>& mR, float dim, Mat<U>& mRinv) {

    for (int j = 0; j < dim; j++) {
        mRinv(j,j) = 1 / mR(j,j);

        for (int i = 0; i < (j); i++) {
            for (int k = 0; k < (j); k++) {
                mRinv(i,j) += mRinv(i,k) * mR(k,j);
            }
        }
        for(int m=0; m<(j); m++) {
            mRinv(m,j) /= -mR(j,j);
        }
    } 
}

template <class T, class U>
void inv6x6(const Mat<T>& A, Mat<U>& Ainv) 
{

  float D_ [36] = {};
  float LT_ [36] = {};
  float LTinv_ [36] = {};

  Mat<float> D(&D_[0], 6, 6);
  Mat<float> LT(&LT_[0], 6, 6);
  Mat<float> LTinv(&LTinv_[0], 6, 6);

  // Build D and upper triangular LT
  for (unsigned j = 0; j < 6; ++j) {
    LT(j,j) = 1.0;
    D(j,j) = A(j,j);
    for (unsigned k = 0; k < j; ++k) {
      D(j,j) -= LT(k,j) * LT(k,j) * D(k,k);
    }

    for (unsigned i = j+1; i < 6; ++i) {
      LT(j,i) = (1 / D(j,j)) * A(i,j);
      for (unsigned k = 0; k < j; ++k) {
        LT(j,i) -= (1 / D(j,j)) * LT(k,i) * LT(k,j) * D(k,k);
      }
    }
  }

  // Replace D with D inv now
  Mat<float> Dinv(&D_[0], 6, 6);
  for (unsigned j = 0; j < 6; ++j) {
    Dinv(j,j) = 1 / Dinv(j,j);
  }

  // Invert LT
  inv_uppertriang(LT, 6, LTinv);

  float intMat_ [36] = {};
  Mat<float> intMat(&intMat_[0], 6, 6);

  matMul(LTinv, Dinv, intMat);  
  matMul(intMat, LTinv, Ainv, false, true);
}


template <class T, class U>
void inv9x9(const Mat<T>& A, Mat<U>& Ainv) 
{

  float D_ [81] = {};
  float LT_ [81] = {};
  float LTinv_ [81] = {};

  Mat<float> D(&D_[0], 9, 9);
  Mat<float> LT(&LT_[0], 9, 9);
  Mat<float> LTinv(&LTinv_[0], 9, 9);

  // Build D and upper triangular LT
  for (unsigned j = 0; j < 9; ++j) {
    LT(j,j) = 1.0;
    D(j,j) = A(j,j);
    for (unsigned k = 0; k < j; ++k) {
      D(j,j) -= LT(k,j) * LT(k,j) * D(k,k);
    }

    for (unsigned i = j+1; i < 9; ++i) {
      LT(j,i) = (1 / D(j,j)) * A(i,j);
      for (unsigned k = 0; k < j; ++k) {
        LT(j,i) -= (1 / D(j,j)) * LT(k,i) * LT(k,j) * D(k,k);
      }
    }
  }

  // Replace D with D inv now
  Mat<float> Dinv(&D_[0], 9, 9);
  for (unsigned j = 0; j < 9; ++j) {
    Dinv(j,j) = 1 / Dinv(j,j);
  }

  // Invert LT
  inv_uppertriang(LT, 9, LTinv);

  float intMat_ [81] = {};
  Mat<float> intMat(&intMat_[0], 9, 9);

  matMul(LTinv, Dinv, intMat);  
  matMul(intMat, LTinv, Ainv, false, true);
}