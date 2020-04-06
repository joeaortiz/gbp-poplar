// Compute the parameters of the Gaussian distribution in the mean form from the information form parameters
template <class T, class U, class V, class W>
void inf2mean6x6(const Mat<T>& eta, const Mat<U>& lambda, Mat<V>& mean, Mat<W>& sigma)
{
    inv6x6(lambda, sigma);
    matMul(sigma, eta, mean);
}

// Compute the parameters of the Gaussian distribution in the mean form from the information form parameters
template <class T, class U, class V, class W>
void inf2mean3x3(const Mat<T>& eta, const Mat<U>& lambda, Mat<V>& mean, Mat<W>& sigma)
{
    inv3x3(lambda, sigma);
    matMul(sigma, eta, mean);
}


// Hat operator for so(3). (Same as cross product)
template <class T, class U>
void so3_hat_operator(const Mat<T>& v, Mat<U>& v_hat) 
{
  v_hat(0,1) = - v(2,0);
  v_hat(0,2) = v(1,0);
  v_hat(1,0) = v(2,0);
  v_hat(1,2) = - v(0,0);
  v_hat(2,0) = - v(1,0);
  v_hat(2,1) = v(0,0);
}

// Maps so(3) to SO(3)
template <class T, class U>
void so3exp(const Mat<T>& v, Mat<U>& R) 
{
  R(0,0) = 1.f;
  R(1,1) = 1.f;
  R(2,2) = 1.f;
  float theta = std::sqrt(v(0,0) * v(0,0) + v(1,0) * v(1,0) + v(2,0) * v(2,0));
  if (theta > 1e-6f) {
    float sintheta = std::sin(theta);
    float costheta = std::cos(theta);
    float v_hat_ [9] = {};
    float v_hat_sqr_ [9] = {};
    Mat<float> v_hat(&v_hat_[0], 3, 3);
    Mat<float> v_hat_sqr(&v_hat_sqr_[0], 3, 3);
    so3_hat_operator(v, v_hat);
    matMul(v_hat, v_hat, v_hat_sqr);

    for (unsigned i = 0; i < 3; ++i) {
      for (unsigned j = 0; j < 3; ++j) {
        R(i,j) += (sintheta / theta) * v_hat(i,j);
        R(i,j) += ((1 - costheta) / (theta * theta)) * v_hat_sqr(i,j);
      }
    }
  }
}

// Get transformation from 6 parameters
template <class V, class U>
void tranf_w2c(const Mat<V>& x, Mat<U>& Tw2c) 
{
  Tw2c(3,3) = 0.f;
  Tw2c(0,3) = x(0,0);
  Tw2c(1,3) = x(1,0);
  Tw2c(2,3) = x(2,0);

  float v_ [3] = {};
  float R_ [9] = {};
  Mat<float> v(&v_[0], 3, 1);
  Mat<float> R(&R_[0], 3, 3);
  for (unsigned i = 0; i < 3; ++i) {
    v(i,0) = x(i+3,0);
  }
  so3exp(v, R);

  for (unsigned i = 0; i < 3; ++i) {
    for (unsigned j = 0; j < 3; ++j) {
      Tw2c(i,j) = R(i,j); 
    }
  }
}

template <class T, class U, class V, class W>
void hfunc(const Mat<T>& cam_params, const Mat<U>& lmk_loc, Mat<W>& K, Mat<V>& hx)
{
  float Tw2c_ [16] = {};
  Mat<float> Tw2c(&Tw2c_[0], 4, 4);

  tranf_w2c(cam_params, Tw2c);

  float b2 [4] = {};
  Mat<float> yhomog(b2, 4, 1);
  yhomog(0,0) = lmk_loc(0,0);
  yhomog(1,0) = lmk_loc(1,0);
  yhomog(2,0) = lmk_loc(2,0);
  yhomog(3,0) = 1.0;

  float y_cf_ [4] = {};
  Mat<float> y_cf(&y_cf_[0], 4, 1);
  matMul(Tw2c, yhomog, y_cf);

  hx(0,0) = K(0,0) * (y_cf(0,0) / y_cf(2,0)) + K(0,2);
  hx(1,0) = K(1,1) * (y_cf(1,0) / y_cf(2,0)) + K(1,2); 
}


template <class T, class U, class V, class W, class X>
void Jac(const Mat<T>& kf_lin_point, const Mat<U>& lmk_lin_point, const Mat<V>& K, Mat<W>& Jkf, Mat<X>& Jlmk) 
{
  float Tw2c_ [16] = {};
  float R_ [9] = {};
  Mat<float> Tw2c(&Tw2c_[0], 4, 4);
  Mat<float> R(&R_[0], 3, 3);

  tranf_w2c(kf_lin_point, Tw2c);

  for (unsigned i = 0; i < 3; ++i) {
    for (unsigned j = 0; j < 3; ++j) {
      R(i,j) = Tw2c(i,j);
    }
  }

  float b2 [4] = {};
  Mat<float> yhomog(b2, 4, 1);
  yhomog(0,0) = lmk_lin_point(0,0);
  yhomog(1,0) = lmk_lin_point(1,0);
  yhomog(2,0) = lmk_lin_point(2,0);
  yhomog(3,0) = 1.0;

  float y_cf_ [4] = {};
  Mat<float> y_cf(&y_cf_[0], 4, 1);
  matMul(Tw2c, yhomog, y_cf);


  float J_proj_ [6] = {};
  Mat<float> J_proj(J_proj_, 2, 3);

  J_proj(0,0) = K(0,0) / y_cf(2,0);
  J_proj(0,2) = - (K(0,0) * y_cf(0,0)) / (y_cf(2,0) * y_cf(2,0));
  J_proj(1,1) = K(1,1) / y_cf(2,0);
  J_proj(1,2) = - (K(1,1) * y_cf(1,0)) / (y_cf(2,0) * y_cf(2,0));

  // Landmark Jacobian
  matMul(J_proj, R, Jlmk);

  // Translation part of Keyframe Jacobian
  for (unsigned i = 0; i < 2; ++i) {
    for (unsigned j = 0; j < 3; ++j) {
      Jkf(i,j) = J_proj(i,j);
    }
  }
  // Rotation part of Keyframe Jacobian
  float v_ [3] = {}; 
  float dRydv_ [9] = {};
  float Jkf_rot_ [6] = {};
  Mat<float> v(&v_[0], 3, 1);  // rotation parameters
  Mat<float> dRydv(&dRydv_[0], 3, 3);
  Mat<float> Jkf_rot(&Jkf_rot_[0], 2, 3);
  for (unsigned i = 0; i < 3; ++i) {
    v(i,0) = kf_lin_point(i+3, 0);
  }

  float v_hat_ [9] = {};
  float ywf_hat_ [9] = {};
  float vv_outer_ [9] = {};
  float RT_minus_I_ [9] = {};
  float R_ywfhat_ [9] = {};
  float numerator_ [9] = {};
  float denominator = 0.0;

  Mat<float> v_hat(v_hat_, 3, 3);
  Mat<float> ywf_hat(ywf_hat_, 3, 3);
  Mat<float> vv_outer(vv_outer_, 3, 3);
  Mat<float> RT_minus_I(RT_minus_I_, 3, 3);
  Mat<float> R_ywfhat(R_ywfhat_, 3, 3);
  Mat<float> numerator(numerator_, 3, 3);


  so3_hat_operator(v, v_hat);
  so3_hat_operator(lmk_lin_point, ywf_hat);

  outerProduct(v, v, vv_outer);
  for (unsigned i = 0; i < 3; ++i) {
    RT_minus_I(i,i) = - 1.f;
    for (unsigned j = 0; j < 3; ++j) {
      RT_minus_I(i,j) += R(j,i);
    }
  }

  matMul(R, ywf_hat, R_ywfhat);
  matMul(RT_minus_I, v_hat, numerator);
  for (unsigned i = 0; i < 3; ++i) {
    for (unsigned j = 0; j < 3; ++j) {
      numerator(i,j) += vv_outer(i,j);
    }
  }

  denominator = dotProduct(v, v);

  matMul(R_ywfhat, numerator, dRydv);
  for (unsigned i = 0; i < 3; ++i) {
    for (unsigned j = 0; j < 3; ++j) {
      dRydv(i,j) = - dRydv(i,j) / denominator;
    }
  }

  matMul(J_proj, dRydv, Jkf_rot);
  for (unsigned i = 0; i < 2; ++i) {
    for (unsigned j = 0; j < 3; ++j) {
      Jkf(i,j+3) = Jkf_rot(i,j);
    }
  }

}
