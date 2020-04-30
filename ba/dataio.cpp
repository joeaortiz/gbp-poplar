#include "../include/dataio.h"
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <random>
#include <chrono>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include "../include/util.h"

using namespace std;
using namespace Eigen;


bool BALProblem::LoadFile(const char* filename) {
  FILE* fptr = fopen(filename, "r");
  if (fptr == NULL) {
    return false;
  };

  FscanfOrDie(fptr, "%d", &n_keyframes_);
  FscanfOrDie(fptr, "%d", &n_points_);
  FscanfOrDie(fptr, "%d", &n_edges_);

  point_index_ = new unsigned[n_edges_];
  camera_index_ = new unsigned[n_edges_];
  observations_ = new double[2 * n_edges_];

  num_parameters_ = 6 * n_keyframes_ + 3 * n_points_;
  parameters_ = new double[num_parameters_];

  fx_ = new double[1];
  fy_ = new double[1];
  cx_ = new double[1];
  cy_ = new double[1];
  FscanfOrDie(fptr, "%lf", fx_);
  FscanfOrDie(fptr, "%lf", fy_);
  FscanfOrDie(fptr, "%lf", cx_);
  FscanfOrDie(fptr, "%lf", cy_);

  for (int i = 0; i < n_edges_; ++i) {
    FscanfOrDie(fptr, "%d", camera_index_ + i);
    FscanfOrDie(fptr, "%d", point_index_ + i);
    for (int j = 0; j < 2; ++j) {
      FscanfOrDie(fptr, "%lf", observations_ + 2*i + j);
    }
  }

  for (int i = 0; i < num_parameters_; ++i) {
    FscanfOrDie(fptr, "%lf", parameters_ + i);
  }

  fclose(fptr);
  return true;
}

template<typename T>
void BALProblem::FscanfOrDie(FILE *fptr, const char *format, T *value) {
  int num_scanned = fscanf(fptr, format, value);
  if (num_scanned != 1) {
    cout << "Invalid UW data file.";
  }
}

void set_prior_lambda(BALProblem& bal_problem, std::vector<float> K, 
                      float reproj_meas_var, 
                      std::vector<float> cam_priors_mean_,
                      std::vector<float>& cam_priors_eta_,
                      std::vector<float>& cam_priors_lambda_,
                      std::vector<float> lmk_priors_mean_,
                      std::vector<float>& lmk_priors_eta_,
                      std::vector<float>& lmk_priors_lambda_)
{
  for (int c = 0; c < bal_problem.n_keyframes(); ++c) {
    float max_jac = 0.;
    for (int i = 0; i < bal_problem.n_edges(); ++i) {
      Matrix<double,6,1> cam = Map<Matrix<double,6,1>>(bal_problem.camera(c));
      if (bal_problem.camera_index(i) == c) {
        Vector3d lmk = Map<Vector3d>(bal_problem.mutable_point_for_observation(i));

        MatrixXf jac = reprojectionJacFn(cam.cast <float> (), lmk.cast <float> (), K);
        float this_max = jac.cwiseAbs().maxCoeff();
        max_jac = max(max_jac, this_max);
      }
    }
    float lam_scaled = pow(max_jac, 2) / reproj_meas_var;
    Matrix<float,6,1> prior_mean = Map<Matrix<float,6,1>>(&cam_priors_mean_[c*6]);
    Matrix<float,6,1> prior_eta = prior_mean * lam_scaled;
    for (int i = 0; i < 6; ++i) {
      cam_priors_eta_[6*c + i] = prior_eta(i);
      cam_priors_lambda_[36*c + i*6 + i] = lam_scaled;
    }
  }
  
  for (int c = 0; c < bal_problem.n_points(); ++c) {
    float max_jac = 0.;
    for (int i = 0; i < bal_problem.n_edges(); ++i) {
      Vector3d lmk = Map<Vector3d>(bal_problem.point(c));
      if (bal_problem.point_index(i) == c) {
        Matrix<double,6,1> cam = Map<Matrix<double,6,1>>(bal_problem.mutable_camera_for_observation(i));

        MatrixXf jac = reprojectionJacFn(cam.cast <float> (), lmk.cast <float> (), K);
        float this_max = jac.cwiseAbs().maxCoeff();
        max_jac = max(max_jac, this_max);
      }
    }
    float lam_scaled = pow(max_jac, 2) / reproj_meas_var;
    Vector3f prior_mean = Map<Vector3f>(&lmk_priors_mean_[c*3]);
    Vector3f prior_eta = prior_mean * lam_scaled;
    for (int i = 0; i < 3; ++i) {
      lmk_priors_eta_[3*c + i] = prior_eta(i);
      lmk_priors_lambda_[9*c + i*3 + i] = lam_scaled;
    }
  }
}

// Functions to load data 
static unsigned readUnsignedInt(string fname) {

  ifstream input (fname);
  unsigned num;
  input >> num;

  return num;
}

vector<unsigned> readUnsignedIntVector(string fname) {

  string line;
  ifstream myfile (fname);
  vector<unsigned> vec;
  if (myfile.is_open())
  {
    while ( getline (myfile,line) )
    {
      vec.push_back(stoul(line));
    }
    myfile.close();
  }
  else cout << "Unable to open file " << fname << " !!!! \n"; 

  return vec;
}

vector<float> readFloatVector(string fname) {

  string line;
  ifstream myfile (fname);
  vector<float> vec;
  if (myfile.is_open())
  {
    while ( getline (myfile,line) )
    {
      vec.push_back(stof(line));
    }
    myfile.close();
  }
  else cout << "Unable to open file " << fname << " !!!! \n"; 

  return vec;
}

void saveFloatVector(string fname, float* vec, unsigned len) {
  ofstream f;
  f.open(fname);

  for(unsigned j=0; j<len; j++)
  {
      f << fixed << setprecision(7) << vec[j] << endl;
  }

  f.close();
}

void saveBeliefs(string dir, unsigned itr, unsigned n_keyframes, 
                unsigned n_points, unsigned cam_dofs, float* cam_beliefs_eta_, 
                float* cam_beliefs_lambda_, float* lmk_beliefs_eta_, 
                float* lmk_beliefs_lambda_) 
{
  string fce = dir + "/beliefs/cb_eta" + std::to_string(itr) + ".txt";
  string fcl = dir + "/beliefs/cb_lambda" + std::to_string(itr) + ".txt";
  string fle = dir + "/beliefs/lb_eta" + std::to_string(itr) + ".txt";
  string fll = dir + "/beliefs/lb_lambda" + std::to_string(itr) + ".txt";
  saveFloatVector(fce, &cam_beliefs_eta_[0], cam_dofs * n_keyframes);
  saveFloatVector(fcl, &cam_beliefs_lambda_[0], cam_dofs * cam_dofs * n_keyframes);
  saveFloatVector(fle, &lmk_beliefs_eta_[0], 3 * n_points);
  saveFloatVector(fll, &lmk_beliefs_lambda_[0], 9 * n_points);
}


void save_cam_means(ofstream &cmeans_file, int itr, float* cam_beliefs_eta_, 
                    float* cam_beliefs_lambda_, unsigned n_keyframes, 
                    unsigned* cam_active_flag_) {
  float cam_means_ [n_keyframes * 6] = {};

  // Use as many threads as there are cores on the CPU:
  tbb::task_scheduler_init init(tbb::task_scheduler_init::automatic);

  tbb::parallel_for(0U, n_keyframes, [&](unsigned cID) {
    if (cam_active_flag_[cID] == 5) {

      Matrix<float,6,1> cam_eta = Map<Matrix<float,6,1>>(&cam_beliefs_eta_[cID * 6]);
      Matrix<float,6,6> cam_lam = Map<Matrix<float,6,6>>(&cam_beliefs_lambda_[cID * 36]);
      VectorXf cam_mu = cam_lam.transpose().inverse() * cam_eta;

      for (unsigned i = 0; i < 6; ++i) {
        cam_means_[cID * 6 + i] = cam_mu(i);
      }
    }
  });

  cmeans_file << "Iteration " << itr << "\n";
  for (unsigned cID = 0; cID < n_keyframes; ++cID) {
    if (cam_active_flag_[cID] == 5) {
      for (unsigned i = 0; i < 6; ++i) {
        cmeans_file << fixed << setprecision(7) << cam_means_[cID * 6 + i] << "\n";
      }
    }
  }

}

void save_lmk_means(ofstream &lmeans_file, int itr, float* lmk_beliefs_eta_, 
                    float* lmk_beliefs_lambda_, unsigned n_points, 
                    unsigned* lmk_active_flag_) {
  float lmk_means_ [n_points * 3] = {};

  // Use as many threads as there are cores on the CPU:
  tbb::task_scheduler_init init(tbb::task_scheduler_init::automatic);

  tbb::parallel_for(0U, n_points, [&](unsigned lID) {
    if (lmk_active_flag_[lID] == 5) {

      Vector3f lmk_eta = Map<Vector3f>(&lmk_beliefs_eta_[lID * 3]);
      Matrix3f lmk_lam = Map<Matrix3f>(&lmk_beliefs_lambda_[lID * 9]);
      Vector3f lmk_mu = lmk_lam.transpose().inverse() * lmk_eta;

      for (unsigned i = 0; i < 3; ++i) {
        lmk_means_[lID * 3 + i] = lmk_mu(i);
      }
    }
  });

  lmeans_file << "Iteration " << itr << "\n";
  for (unsigned lID = 0; lID < n_points; ++lID) {
    if (lmk_active_flag_[lID] == 5) {
      for (unsigned i = 0; i < 3; ++i) {
        lmeans_file << fixed << setprecision(7) << lmk_means_[lID * 3 + i] << "\n";
      }
    }
  }

}

void save_message_KL(ofstream &kl_file, int itr, unsigned n_edges, 
                    std::vector<unsigned int> active_flag,
                    unsigned* measurements_camIDs, unsigned* measurements_lIDs, 
                    int data_counter, unsigned max_nkfedges, unsigned max_nlmkedges, 
                    unsigned cam_dofs, float* cam_messages_lambda_, 
                    float* lmk_messages_lambda_)
{
  // float* cam_messages_eta_, float* cam_messages_lambda_, float* lmk_messages_eta_, float* lmk_messages_lambda_,
  // float* pcam_messages_eta_, float* pcam_messages_lambda_, float* plmk_messages_eta_, float* plmk_messages_lambda_) {


  // Use as many threads as there are cores on the CPU:
  tbb::task_scheduler_init init(tbb::task_scheduler_init::automatic);

  unsigned n_active_edges = 0;
  for (unsigned e = 0; e < n_edges; ++e) {
    n_active_edges += active_flag[data_counter * n_edges +e];
  }

  float cam_norm [n_active_edges] = {};
  float lmk_norm [n_active_edges] = {};

  tbb::parallel_for(0U, n_active_edges, [&](unsigned e) {
  // for (unsigned e = 0; e < n_active_edges; ++e) {


    unsigned adj_cID = measurements_camIDs[e];
    unsigned adj_lID = measurements_lIDs[e];

    unsigned edges_c = 0;
    unsigned edges_l = 0;
    for (unsigned i = 0; i < e; ++i) {
      if (measurements_camIDs[i] == adj_cID) {
        edges_c += 1;
      }
    }
    unsigned count_ledges_l = 0;
    for (unsigned i = 0; i < e; ++i) {
      if (measurements_lIDs[i] == adj_lID) {
        edges_l += 1;
      }
    }

    // Matrix<float,6,1> cam_mess_eta = Map<Matrix<float,6,1>>(&cam_messages_eta_[(max_nkfedges + 1) * cam_dofs * adj_cID + cam_dofs * (edges_c + 1)]);
    Matrix<float,6,6> cam_mess_lambda = Map<Matrix<float,6,6>>(&cam_messages_lambda_[(max_nkfedges + 1) * cam_dofs * cam_dofs * adj_cID + cam_dofs * cam_dofs * (edges_c + 1)]);
    // Matrix<float,6,1> pcam_mess_eta = Map<Matrix<float,6,1>>(&pcam_messages_eta_[(max_nkfedges + 1) * cam_dofs * adj_cID + cam_dofs * (edges_c + 1)]);
    // Matrix<float,6,6> pcam_mess_lambda = Map<Matrix<float,6,6>>(&pcam_messages_lambda_[(max_nkfedges + 1) * cam_dofs * cam_dofs * adj_cID + cam_dofs * cam_dofs * (edges_c + 1)]);

    // Matrix<float,3,1> lmk_mess_eta = Map<Matrix<float,3,1>>(&lmk_messages_eta_[(max_nlmkedges + 1) * 3 * adj_lID + 3 * (edges_l + 1)]);
    Matrix<float,3,3> lmk_mess_lambda = Map<Matrix<float,3,3>>(&lmk_messages_lambda_[(max_nlmkedges + 1) * 9 * adj_lID + 9 * (edges_l + 1)]);
    // Matrix<float,3,1> plmk_mess_eta = Map<Matrix<float,3,1>>(&plmk_messages_eta_[(max_nlmkedges + 1) * 3 * adj_lID + 3 * (edges_l + 1)]);
    // Matrix<float,3,3> plmk_mess_lambda = Map<Matrix<float,3,3>>(&plmk_messages_lambda_[(max_nlmkedges + 1) * 9 * adj_lID + 9 * (edges_l + 1)]);

    // camKL[e] = symmetricKL(cam_mess_eta, pcam_mess_eta, cam_mess_lambda, pcam_mess_lambda);
    // lmkKL[e] = symmetricKL(lmk_mess_eta, plmk_mess_eta, lmk_mess_lambda, plmk_mess_lambda);
    cam_norm[e] = cam_mess_lambda.squaredNorm();
    lmk_norm[e] = lmk_mess_lambda.squaredNorm();

  });

  kl_file << "Iteration " << itr << "\n";
  float cam_KL_tot = 0.0;
  float lmk_KL_tot = 0.0;
  for (unsigned e = 0; e < n_active_edges; ++e) {
    kl_file << cam_norm[e] << " " << lmk_norm[e] << "\n";
    cam_KL_tot += cam_norm[e];
    lmk_KL_tot += lmk_norm[e];
  }

  cout << "Average cam norm " << cam_KL_tot / n_active_edges << "  Average lmk norm " << lmk_KL_tot / n_active_edges << "\n";
}


void add_cam_trans_noise(std::vector<float>& cam_priors_mean_, unsigned n_keyframes,
                        float cam_trans_noise_std, unsigned k)
{
  std::cout << "\nAdding Gaussian noise with std: " << cam_trans_noise_std << "m to the keyframe translaton intialisations\n";
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator (seed);
  std::normal_distribution<float> trans_noise(0.f, cam_trans_noise_std);

  for (unsigned cID = k; cID < n_keyframes; ++cID) {
    for (unsigned i = 0; i < 3; ++i) {
      cam_priors_mean_[cID * 6 + i] += trans_noise(generator);
    }
  }
}

void add_cam_rot_noise(std::vector<float>& cam_priors_mean_, unsigned n_keyframes,
                        float cam_rot_noise_std, unsigned k)
{
  std::cout << "Adding Gaussian noise with std: " << cam_rot_noise_std << " to the keyframe rotation intialisations\n";
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator (seed);
  std::normal_distribution<float> rot_noise(0.f, cam_rot_noise_std);

  for (unsigned cID = k; cID < n_keyframes; ++cID) {

    float rot_angle_radians = rot_noise(generator) * M_PI / 180;
    int axis = (std::rand()%3);

    Matrix3f R;
    if (axis == 0) {
      // Matrix3f R;
      R << 1.f, 0.f, 0.f,
           0.f, cos(rot_angle_radians), -sin(rot_angle_radians),
           0.f, sin(rot_angle_radians), cos(rot_angle_radians);
    }
    else if (axis == 1) {
      // Matrix3f R;
      R << cos(rot_angle_radians), 0.f, sin(rot_angle_radians),
           0.f, 1.f, 0.f,
           -sin(rot_angle_radians), 0.f, cos(rot_angle_radians);
    }
    else if (axis == 2) {
      // Matrix3f R;
      R << cos(rot_angle_radians), -sin(rot_angle_radians), 0.f,
           sin(rot_angle_radians), cos(rot_angle_radians), 0.f,
           0.f, 0.f, 1.f;
    }

    Vector3f rot_component;
    rot_component << cam_priors_mean_[cID*6 + 3], cam_priors_mean_[cID*6 + 4], cam_priors_mean_[cID*6 + 5];
    Matrix3f Rw2c = eigenso3exp(rot_component);
    Matrix4f Tw2c;
    Tw2c << Rw2c(0,0), Rw2c(0,1), Rw2c(0,2), cam_priors_mean_[cID*6],
            Rw2c(1,0), Rw2c(1,1), Rw2c(1,2), cam_priors_mean_[cID*6 + 1],
            Rw2c(2,0), Rw2c(2,1), Rw2c(2,2), cam_priors_mean_[cID*6 + 2],
            0.0, 0.0, 0.0, 1.0;
    Matrix4f Tc2w = Tw2c.inverse();
    Tc2w.block<3,3>(0,0) = R * Tc2w.block<3,3>(0,0);
    Tw2c = Tc2w.inverse();

    cam_priors_mean_[cID*6] = Tw2c(0,3);
    cam_priors_mean_[cID*6 + 1] = Tw2c(1,3);
    cam_priors_mean_[cID*6 + 2] = Tw2c(2,3);

    Rw2c = Tw2c.block<3,3>(0,0);
    Vector3f w = so3log(Rw2c);
    cam_priors_mean_[cID*6 + 3] = w(0);
    cam_priors_mean_[cID*6 + 4] = w(1);
    cam_priors_mean_[cID*6 + 5] = w(2);
  }
}

void add_lmk_noise(std::vector<float>& lmk_priors_mean_, unsigned n_points, 
                          float lmk_trans_noise_std)
{  
  std::cout << "Adding Gaussian noise with std: " << lmk_trans_noise_std << "m to the landmark intialisations\n";
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator (seed);
  std::normal_distribution<float> noise(0.f, lmk_trans_noise_std);

  for (unsigned lID = 0; lID < n_points; ++lID) {
    for (unsigned i = 0; i < 3; ++i) {
      lmk_priors_mean_[lID * 3 + i] += noise(generator);
    }
  }
}

void av_depth_init(float av_depth, unsigned n_keyframes, unsigned n_edges, unsigned n_points, 
                  std::vector<unsigned> measurements_camIDs, 
                  std::vector<unsigned> measurements_lIDs,
                  std::vector<float> cam_priors_mean_, 
                  std::vector<float>& lmk_priors_mean_) 
{
  std::cout << "Initialising all landmarks at an average depth of: " << av_depth << "\n";
  std::vector<unsigned> lmk_done(n_points, 0);

  for (unsigned cID = 0; cID < n_keyframes; ++cID) {
    Vector3f rot_component;
    rot_component << cam_priors_mean_[cID*6 + 3], cam_priors_mean_[cID*6 + 4], cam_priors_mean_[cID*6 + 5];
    Matrix3f Rw2c = eigenso3exp(rot_component);
    Matrix4f Tw2c;
    Tw2c << Rw2c(0,0), Rw2c(0,1), Rw2c(0,2), cam_priors_mean_[cID*6],
            Rw2c(1,0), Rw2c(1,1), Rw2c(1,2), cam_priors_mean_[cID*6 + 1],
            Rw2c(2,0), Rw2c(2,1), Rw2c(2,2), cam_priors_mean_[cID*6 + 2],
            0.0, 0.0, 0.0, 1.0;

    Vector4f loc_cam_frame;
    loc_cam_frame << 0.0, 0.0, 1.0, 1.0;
    Vector4f new_lmk_mu_wf_homog = Tw2c.inverse() * loc_cam_frame;
    Vector3f new_lmk_mu_wf;
    new_lmk_mu_wf << new_lmk_mu_wf_homog(0), new_lmk_mu_wf_homog(1), new_lmk_mu_wf_homog(2);

    for (unsigned e = 0; e < n_edges; ++e) {
      if (measurements_camIDs[e] == cID) {
        if (lmk_done[measurements_lIDs[e]] == 0) {
          lmk_priors_mean_[measurements_lIDs[e] * 3] = new_lmk_mu_wf(0);
          lmk_priors_mean_[measurements_lIDs[e] * 3 + 1] = new_lmk_mu_wf(1);
          lmk_priors_mean_[measurements_lIDs[e] * 3 + 2] = new_lmk_mu_wf(2);
          lmk_done[measurements_lIDs[e]] = 1;
        }
      }
    }
  }
}

void create_flags(BALProblem& bal_problem, std::vector<unsigned int>& active_flag_, 
                  std::vector<unsigned int>& cam_weaken_flag_, 
                  std::vector<unsigned int>& lmk_weaken_flag_,
                  std::vector<unsigned int>& lmk_active_flag, int steps)
{  
  // Begin with first 2 keyframes
  cam_weaken_flag_[0] = steps;  
  cam_weaken_flag_[1] = steps;

  for (int i = 0; i < bal_problem.n_edges(); ++i) {
    if ((bal_problem.camera_index(i) == 0) || (bal_problem.camera_index(i) == 1)) {
      active_flag_[i] = 1;
      lmk_weaken_flag_[bal_problem.point_index(i)] = steps;
    }
  }

  for (int i=0; i<lmk_weaken_flag_.size(); i++) {
    lmk_active_flag[i] = lmk_weaken_flag_[i]; 
  }

}

int update_flags(BALProblem& bal_problem, std::vector<unsigned int>& active_flag_, 
             std::vector<unsigned int>& lmk_weaken_flag_, 
             std::vector<unsigned int>& cam_weaken_flag_, 
             std::vector<unsigned int>& lmk_active_flag, 
             unsigned steps, unsigned data_counter)
{
  for (int i = 0; i < bal_problem.n_edges(); ++i) {
    if (bal_problem.camera_index(i) == data_counter + 1) {
      active_flag_[i] = 1;
    }
    if (bal_problem.camera_index(i) <= data_counter + 1) {
      lmk_weaken_flag_[bal_problem.point_index(i)] = steps;
    }
  }

  for (int c=0; c < bal_problem.n_keyframes(); ++c) {
    cam_weaken_flag_[c] = 0;
  }
  cam_weaken_flag_[data_counter+1] = steps;

  for (int c = 0; c < bal_problem.n_points(); ++c) {
    lmk_weaken_flag_[c] -= lmk_active_flag[c];
    lmk_active_flag[c] += lmk_weaken_flag_[c];
  }

  int n_new_lmks = 0;
  for (unsigned i = 0; i < bal_problem.n_points(); ++i) {
    n_new_lmks += lmk_weaken_flag_[i];
  }
  n_new_lmks /= steps;
  return n_new_lmks;
}

