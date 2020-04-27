#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

#ifndef BALPROBLEM_H
#define BALPROBLEM_H
// Read a Bundle Adjustment dataset.
class BALProblem 
{
 public:
  ~BALProblem() {
    delete[] point_index_;
    delete[] camera_index_;
    delete[] observations_;
    delete[] fx_;
    delete[] fy_;
    delete[] cx_;
    delete[] cy_;
    delete[] parameters_;
  }

  unsigned n_edges()       const { return n_edges_;               }
  unsigned n_keyframes()       const { return n_keyframes_;               }
  unsigned n_points()       const { return n_points_;               }
  unsigned camera_index(int i)       const { return camera_index_[i];               }
  unsigned point_index(int i)       const { return point_index_[i];               }

  const double* observations() const { return observations_;                   }
  const double* fx() const { return fx_;                   }
  const double* fy() const { return fy_;                   }
  const double* cx() const { return cx_;                   }
  const double* cy() const { return cy_;                   }
  double* mutable_cameras()          { return parameters_;                     }
  double* mutable_points()           { return parameters_  + 6 * n_keyframes_; }
  
  double* camera(int i)          { return mutable_cameras() + i * 6;                     }
  double* point(int i)           { return mutable_points() + i * 3; }

  double* mutable_camera_for_observation(int i) {
    return mutable_cameras() + camera_index_[i] * 6;
  }
  double* mutable_point_for_observation(int i) {
    return mutable_points() + point_index_[i] * 3;
  }

  bool LoadFile(const char* filename);

 private:
  template<typename T>
  void FscanfOrDie(FILE *fptr, const char *format, T *value);

  unsigned n_keyframes_;
  unsigned n_points_;
  unsigned n_edges_;
  int num_parameters_;
  double* fx_;
  double* fy_;
  double* cx_;
  double* cy_;

  unsigned* point_index_;
  unsigned* camera_index_;
  double* observations_;
  double* parameters_;
};
#endif

// Sets the lambda of the priors to be of the same order of magnitude as 
// the lambda for the reprojection for numerical stability in the first few 
// iterations. The priors are subsequently weakened
void set_prior_lambda(BALProblem& bal_problem, std::vector<float> K, 
                      float reproj_meas_var, 
                      std::vector<float> cam_priors_mean_,
                      std::vector<float>& cam_priors_eta_,
                      std::vector<float>& cam_priors_lambda_,
                      std::vector<float> lmk_priors_mean_,
                      std::vector<float>& lmk_priors_eta_,
                      std::vector<float>& lmk_priors_lambda_);

// Functions to load data 
static unsigned readUnsignedInt(std::string fname);

std::vector<unsigned> readUnsignedIntVector(std::string fname);

std::vector<float> readFloatVector(std::string fname);

void saveFloatVector(std::string fname, float* vec, unsigned len);

void saveBeliefs(std::string dir, unsigned itr, unsigned n_keyframes, 
                unsigned n_points, unsigned cam_dofs, float* cam_beliefs_eta_, 
                float* cam_beliefs_lambda_, float* lmk_beliefs_eta_, 
                float* lmk_beliefs_lambda_);

void save_cam_means(std::ofstream &cmeans_file, int itr, float* cam_beliefs_eta_, 
                    float* cam_beliefs_lambda_, unsigned n_keyframes, 
                    unsigned* cam_active_flag_);

void save_lmk_means(std::ofstream &lmeans_file, int itr, float* lmk_beliefs_eta_, 
                    float* lmk_beliefs_lambda_, unsigned n_points, 
                    unsigned* lmk_active_flag_);

void save_message_KL(std::ofstream &kl_file, int itr, unsigned n_edges, 
                    std::vector<unsigned int> active_flag,
                    unsigned* measurements_camIDs, unsigned* measurements_lIDs, 
                    int data_counter, unsigned max_nkfedges, unsigned max_nlmkedges, 
                    unsigned cam_dofs, float* cam_messages_lambda_, 
                    float* lmk_messages_lambda_);

// Add noise to the translation of the keyframe pose initialisations
// Don't add noise to the first k keyframes which are used to anchor the scale
void add_cam_trans_noise(std::vector<float>& cam_priors_mean_, unsigned n_keyframes,
                        float cam_trans_noise_std, unsigned k=2);

// Add noise to the rotation of the keyframe pose initialisations
void add_cam_rot_noise(std::vector<float>& cam_priors_mean_, unsigned n_keyframes,
                        float cam_rot_noise_std, unsigned k=2);

// Add noise to landmark prior means
void add_lmk_noise(std::vector<float>& lmk_priors_mean_, unsigned n_points, 
                          float lmk_trans_noise_std);

// Overwrites lmk_priors_mean_ such that landmarks are placed at an 
// average depth from the first keyframe that observes them
void av_depth_init(float av_depth, unsigned n_keyframes, unsigned n_edges, unsigned n_points,
                  std::vector<unsigned> measurements_camIDs, 
                  std::vector<unsigned> measurements_lIDs,
                  std::vector<float> cam_priors_mean_, 
                  std::vector<float>& lmk_priors_mean_);
