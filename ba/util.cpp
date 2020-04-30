#include "../include/util.h"
#include <Eigen/Dense>
#include <vector>
#include <iostream>
#include <tbb/tbb.h>

using namespace std;
using namespace Eigen;


Matrix3f eigenso3hat(Vector3f w) 
{
  Matrix3f what;
  what << 0.0, -w(2), w(1),
          w(2), 0.0, -w(0),
          -w(1), w(0), 0.0;
  return what;
}

Matrix3f eigenso3exp(Vector3f w) 
{
  float theta = w.norm();
  Matrix3f R = Matrix3f::Identity();
  if (theta < 1e-6) {
    return R;
  }
  else {
    Matrix3f what = eigenso3hat(w);
    R += (sin(theta) / theta) * what + ((1 - cos(theta)) / (theta * theta)) * (what * what);
    return R;
  }
}

Vector3f so3log(Matrix3f R) 
{
  float d = 0.5 * (R.trace() - 1);

  Matrix3f lnR = (acos(d) / (2 * sqrt(1 - d*d))) * (R - R.transpose());

  Vector3f w;
  w(0) = lnR(2,1);
  w(1) = lnR(0,2);
  w(2) = lnR(1,0);

  return w;
}

MatrixXf reprojectionJacFn(VectorXf cam, Vector3f lmk, std::vector<float> K_vec) 
{
  Matrix3f K = Map<Matrix<float,3,3>>(K_vec.data()).transpose();
  Vector3f w;
  Vector3f t;
  t << cam(0), cam(1), cam(2);
  w << cam(3), cam(4), cam(5);
  Matrix3f Rcw = eigenso3exp(w);

  Vector3f lmk_cf = Rcw * lmk + t;
  Vector3f p = K * lmk_cf;

  MatrixXf j_proj(2,3);
  j_proj << 1 / p(2), 0, -p(0) / pow(p(2),2),
           0, 1 / p(2), -p(1) / pow(p(2),2);

  Matrix3f dR_wx_dw = - eigenso3hat(Rcw * lmk);

  MatrixXf jac(2, 9); 
  jac.block(0, 0, 2, 3) = j_proj * K;
  jac.block(0, 3, 2, 3) = j_proj * K * dR_wx_dw;
  jac.block(0, 6, 2, 3) = (j_proj * K) * Rcw;

  return jac;
}

void eval_reprojection_error(float* reproj, unsigned n_edges, 
                      vector<unsigned int> active_flag,
                      float* cam_beliefs_eta_, float* cam_beliefs_lambda_, 
                      float* lmk_beliefs_eta_, float* lmk_beliefs_lambda_,
                      unsigned* measurements_camIDs, unsigned* measurements_lIDs, 
                      float* measurements_, float* K_,
                      const vector<unsigned int>& bad_associations) 
{
  Matrix3f K = Map<Matrix3f>(K_).transpose();
  reproj[0] = 0.0;
  reproj[1] = 0.0;
  unsigned n_active_edges = 0;

  // Make two vectors for storing every reprojection result computed in
  // parallel so that no locking is required in the parallel loop:
  vector<float> reprojNorm(n_edges, 0.f);
  vector<float> reprojSqNorm(n_edges, 0.f);

  // Use as many threads as there are cores on the CPU:
  tbb::task_scheduler_init init(tbb::task_scheduler_init::automatic);

  for (unsigned e = 0; e < n_edges; ++e) {
    n_active_edges += active_flag[e];
  }

  tbb::parallel_for(0U, n_active_edges, [&](unsigned e) {
    if ((find(bad_associations.begin(), bad_associations.end(), e) == bad_associations.end())) {

      Matrix<float,6,1> cam_eta = Map<Matrix<float,6,1>>(&cam_beliefs_eta_[measurements_camIDs[e] * 6]);
      Matrix<float,6,6> cam_lam = Map<Matrix<float,6,6>>(&cam_beliefs_lambda_[measurements_camIDs[e] * 36]);
      VectorXf cam_mu = cam_lam.transpose().inverse() * cam_eta;

      Vector3f lmk_eta = Map<Vector3f>(&lmk_beliefs_eta_[measurements_lIDs[e] * 3]);
      Matrix3f lmk_lam = Map<Matrix3f>(&lmk_beliefs_lambda_[measurements_lIDs[e] * 9]);
      Vector3f lmk_mu = lmk_lam.transpose().inverse() * lmk_eta;

      Vector3f w;
      w << cam_mu(3), cam_mu(4), cam_mu(5);
      Matrix3f R = eigenso3exp(w);

      Vector3f pcf = R * lmk_mu;

      pcf(0) += cam_mu(0);
      pcf(1) += cam_mu(1);
      pcf(2) += cam_mu(2);

      Vector3f predicted = (K * pcf) / pcf(2);
      Vector2f residuals = Map<Vector2f>(&measurements_[2*e]);

      residuals(0) -= predicted(0);
      residuals(1) -= predicted(1);

      reprojNorm[e] = residuals.norm();
      reprojSqNorm[e] = 0.5 * residuals.squaredNorm();
    }
  });

  n_active_edges -= bad_associations.size();

  // Now sum up the results outside of the prallel loop.
  // No need to exclude inactive edges while summing as their entries
  // will have been initialised to 0 and then not updated in the loop above:
  for (unsigned e = 0; e < n_edges; ++e) {
    // n_active_edges += active_flag[e];
    reproj[0] += reprojNorm[e];
    reproj[1] += reprojSqNorm[e];
  }

  // cout << "Number of active edges: " << n_active_edges << "\n";
  reproj[0] /= n_active_edges;
}

void update_eta(unsigned n_keyframes, unsigned n_points,
                std::vector<float> cam_priors_lambda_,
                std::vector<float> cam_priors_mean_,
                std::vector<float>& cam_priors_eta_,
                std::vector<float> lmk_priors_lambda_,
                std::vector<float> lmk_priors_mean_,
                std::vector<float>& lmk_priors_eta_)
{
  for (unsigned cID = 0; cID < n_keyframes; ++cID) {
    Matrix<float,6,6> lambda;
    Matrix<float,6,1> mu;
    lambda = Map<Matrix<float,6,6>>(&cam_priors_lambda_[cID * 36]);
    mu = Map<Matrix<float,6,1>>(&cam_priors_mean_[cID*6]);

    Matrix<float,6,1> eta = lambda * mu;

    cam_priors_eta_[cID*6] = eta(0,0);
    cam_priors_eta_[cID*6 + 1] = eta(1,0);
    cam_priors_eta_[cID*6 + 2] = eta(2,0);
    cam_priors_eta_[cID*6 + 3] = eta(3,0);
    cam_priors_eta_[cID*6 + 4] = eta(4,0);
    cam_priors_eta_[cID*6 + 5] = eta(5,0);

  }
  for (unsigned lID = 0; lID < n_points; ++lID) {
    Matrix3f lambda;
    Vector3f mu;
    lambda = Map<Matrix3f>(&lmk_priors_lambda_[lID*9]);
    mu = Map<Vector3f>(&lmk_priors_mean_[lID*3]);
    Vector3f eta = lambda * mu;

    lmk_priors_eta_[lID*3] = eta(0);
    lmk_priors_eta_[lID*3 + 1] = eta(1);
    lmk_priors_eta_[lID*3 + 2] = eta(2);
  }
}

void initialise_new_kf(std::vector<float>& cam_priors_eta_, std::vector<float>& lmk_priors_eta_,
                       float* cam_beliefs_eta_, float* cam_beliefs_lambda_,
                       std::vector<float> cam_priors_lambda_, std::vector<float> lmk_priors_lambda_,
                       std::vector<unsigned int> lmk_weaken_flag_, 
                       unsigned data_counter, unsigned n_points)
{
  Matrix<float,6,1> previous_kf_eta = Map<Matrix<float,6,1>>(&cam_beliefs_eta_[data_counter * 6]);
  Matrix<float,6,6> previous_kf_lam = Map<Matrix<float,6,6>>(&cam_beliefs_lambda_[data_counter * 36]);
  VectorXf previous_kf_mu = previous_kf_lam.transpose().inverse() * previous_kf_eta;
  Matrix<float,6,6> new_kf_lam = Map<Matrix<float,6,6>>(&cam_priors_lambda_[(data_counter + 1) * 36]);
  VectorXf new_kf_eta = new_kf_lam.transpose() * previous_kf_mu;
  for (unsigned i = 0; i < 6; ++i) {
    cam_priors_eta_[(data_counter + 1) * 6 + i] = new_kf_eta(i);
  }

  // Use prior on keyframe for prior on newly observed landmarks
  Vector3f previous_kf_w;
  previous_kf_w << previous_kf_mu(3), previous_kf_mu(4), previous_kf_mu(5);
  Matrix3f previous_kf_R_w2c = eigenso3exp(previous_kf_w);
  Vector4f loc_cam_frame;
  loc_cam_frame << 0.0, 0.0, 1.0, 1.0;
  Matrix4f Tw2c;
  Tw2c << previous_kf_R_w2c(0,0), previous_kf_R_w2c(0,1), previous_kf_R_w2c(0,2), previous_kf_mu(0),
          previous_kf_R_w2c(1,0), previous_kf_R_w2c(1,1), previous_kf_R_w2c(1,2), previous_kf_mu(1),
          previous_kf_R_w2c(2,0), previous_kf_R_w2c(2,1), previous_kf_R_w2c(2,2), previous_kf_mu(2),
          0.0, 0.0, 0.0, 1.0;
  Vector4f new_lmk_mu_wf_homog = Tw2c.inverse() * loc_cam_frame;
  Vector3f new_lmk_mu_wf;
  new_lmk_mu_wf << new_lmk_mu_wf_homog(0), new_lmk_mu_wf_homog(1), new_lmk_mu_wf_homog(2);
  Matrix3f lmk_prior_lambda;
  Vector3f new_lmk_eta;
  for (unsigned i = 0; i < n_points; ++i) {
    if (lmk_weaken_flag_[data_counter*n_points + i] == 5) {  // newly observed landmark
      lmk_prior_lambda = Map<Matrix3f>(&lmk_priors_lambda_[i*9]);
      new_lmk_eta = lmk_prior_lambda.transpose() * new_lmk_mu_wf;
      for (unsigned j = 0; j < 3; ++j) {
        lmk_priors_eta_[i * 3 + j] = new_lmk_eta(j);
      }
    }
  }
}

float KL_divergence(VectorXf eta1, VectorXf eta2, MatrixXf lambda1, MatrixXf lambda2) {

  VectorXf mu1 = lambda1.inverse() * eta1;
  VectorXf mu2 = lambda2.inverse() * eta2;

  float KL =  0.5 * ( (lambda2 * lambda1.inverse()).trace() + (mu2 - mu1).dot(lambda1 * (mu2 - mu1)) - eta1.size() 
              + log( lambda1.determinant() / lambda2.determinant() ) );

  if (isnan(KL)) {
    cout << "was null" << "\n";
    cout << eta1 << "\n";
    cout << lambda1 << "\n";
    cout << eta2 << "\n";
    cout << lambda2 << "\n";
  }


  return KL;

}

float symmetricKL(VectorXf eta1, VectorXf eta2, MatrixXf lambda1, MatrixXf lambda2) {

  return (KL_divergence(eta1, eta2, lambda1, lambda2) + KL_divergence(eta2, eta1, lambda2, lambda1)) / 2;

}

