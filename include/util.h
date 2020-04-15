#include <Eigen/Dense>
#include <vector>

using namespace Eigen;


Matrix3f eigenso3hat(Vector3f w);

Matrix3f eigenso3exp(Vector3f w);

Vector3f so3log(Matrix3f R);

MatrixXf reprojectionJacFn(VectorXf cam, Vector3f lmk, std::vector<float> K_vec);

void eval_reprojection_error(float* reproj, unsigned n_edges, 
                      std::vector<unsigned int> active_flag, int data_counter,
                      float* cam_beliefs_eta_, float* cam_beliefs_lambda_, 
                      float* lmk_beliefs_eta_, float* lmk_beliefs_lambda_,
                      unsigned* measurements_camIDs, unsigned* measurements_lIDs, 
                      float* measurements_, float* K_,
                      const std::vector<unsigned int>& bad_associations=std::vector<unsigned int>() );

// Update eta priors using mean and lambda priors
void update_eta(unsigned n_keyframes, unsigned n_points,
                std::vector<float> cam_priors_lambda_,
                std::vector<float> cam_priors_mean_,
                std::vector<float>& cam_priors_eta_,
                std::vector<float> lmk_priors_lambda_,
                std::vector<float> lmk_priors_mean_,
                std::vector<float>& lmk_priors_eta_);

float KL_divergence(VectorXf eta1, VectorXf eta2, MatrixXf lambda1, MatrixXf lambda2);

float symmetricKL(VectorXf eta1, VectorXf eta2, MatrixXf lambda1, MatrixXf lambda2);
