// Implementation of BP with single type of variable node with 2 dofs. 

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <poplar/CSRFunctions.hpp>
// #include <poplar/codelets.hpp>
#include <popops/Reduce.hpp>
#include <popops/Operation.hpp>
#include <poplar/ProfileValue.hpp>

#include <Eigen/Dense>

#include <tbb/tbb.h>
#include <tbb/concurrent_vector.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/concurrent_unordered_map.h>

#include <boost/program_options.hpp>

#include <random>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <math.h>
#include <cmath>

using namespace std;
using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace Eigen;


// Functions to load data 
static unsigned readUnsignedInt(std::string fname) {

  std::ifstream input (fname);
  unsigned num;
  input >> num;

  return num;
}

std::vector<unsigned> readUnsignedIntVector(std::string fname) {

  std::string line;
  std::ifstream myfile (fname);
  std::vector<unsigned> vec;
  if (myfile.is_open())
  {
    while ( getline (myfile,line) )
    {
      vec.push_back(std::stoul(line));
    }
    myfile.close();
  }
  else std::cout << "Unable to open file " << fname << " !!!! \n"; 

  return vec;
}

std::vector<float> readFloatVector(std::string fname) {

  std::string line;
  std::ifstream myfile (fname);
  std::vector<float> vec;
  if (myfile.is_open())
  {
    while ( getline (myfile,line) )
    {
      vec.push_back(std::stof(line));
    }
    myfile.close();
  }
  else std::cout << "Unable to open file " << fname << " !!!! \n"; 

  return vec;
}

void saveFloatVector(std::string fname, float* vec, unsigned len) {
  ofstream f;
  f.open(fname);

  for(unsigned j=0; j<len; j++)
  {
      f << std::fixed << std::setprecision(7) << vec[j] << endl;
  }

  f.close();
}


// void saveBeliefs(string dir, unsigned itr, unsigned n_keyframes, unsigned n_points, unsigned cam_dofs,
//                  float* cam_beliefs_eta_, float* cam_beliefs_lambda_, float* lmk_beliefs_eta_, float* lmk_beliefs_lambda_) {
//   string fce = "/mnt/data/jortiz/res/" + dir + "/beliefs/cb_eta" + std::to_string(itr) + ".txt";
//   string fcl = "/mnt/data/jortiz/res/" + dir + "/beliefs/cb_lambda" + std::to_string(itr) + ".txt";
//   string fle = "/mnt/data/jortiz/res/" + dir + "/beliefs/lb_eta" + std::to_string(itr) + ".txt";
//   string fll = "/mnt/data/jortiz/res/" + dir + "/beliefs/lb_lambda" + std::to_string(itr) + ".txt";
//   saveFloatVector(fce, &cam_beliefs_eta_[0], cam_dofs * n_keyframes);
//   saveFloatVector(fcl, &cam_beliefs_lambda_[0], cam_dofs * cam_dofs * n_keyframes);
//   saveFloatVector(fle, &lmk_beliefs_eta_[0], 3 * n_points);
//   saveFloatVector(fll, &lmk_beliefs_lambda_[0], 9 * n_points);
// }


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


void save_message_KL(ofstream &kl_file, int itr, unsigned n_edges, std::vector<unsigned int> active_flag,
  unsigned* measurements_camIDs, unsigned* measurements_lIDs, int data_counter,
  unsigned max_nkfedges, unsigned max_nlmkedges, unsigned cam_dofs, 
  float* cam_messages_lambda_, float* lmk_messages_lambda_) {
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



Matrix3f eigenso3hat(Vector3f w) {
  Matrix3f what;
  what << 0.0, -w(2), w(1),
          w(2), 0.0, -w(0),
          -w(1), w(0), 0.0;
  return what;
}


Matrix3f eigenso3exp(Vector3f w) {
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

Vector3f so3log(Matrix3f R) {
  float d = 0.5 * (R.trace() - 1);

  Matrix3f lnR = (acos(d) / (2 * sqrt(1 - d*d))) * (R - R.transpose());

  Vector3f w;
  w(0) = lnR(2,1);
  w(1) = lnR(0,2);
  w(2) = lnR(1,0);

  return w;
}


void save_cam_means(ofstream &cmeans_file, int itr, float* cam_beliefs_eta_, float* cam_beliefs_lambda_, unsigned n_keyframes, unsigned* cam_active_flag_) {
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
        cmeans_file << std::fixed << std::setprecision(7) << cam_means_[cID * 6 + i] << "\n";
      }
    }
  }

}

void save_lmk_means(ofstream &lmeans_file, int itr, float* lmk_beliefs_eta_, float* lmk_beliefs_lambda_, unsigned n_points, unsigned* lmk_active_flag_) {
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
        lmeans_file << std::fixed << std::setprecision(7) << lmk_means_[lID * 3 + i] << "\n";
      }
    }
  }

}


void eval_reproj_parallel(float* reproj, unsigned n_edges, std::vector<unsigned int> active_flag, int data_counter,
                      float* cam_beliefs_eta_, float* cam_beliefs_lambda_, float* lmk_beliefs_eta_, float* lmk_beliefs_lambda_,
                      unsigned* measurements_camIDs, unsigned* measurements_lIDs, float* measurements_, float* K_,
                      const std::vector<unsigned int>& bad_associations=std::vector<unsigned int>() ) {
  Matrix3f K = Map<Matrix3f>(K_).transpose();
  reproj[0] = 0.0;
  reproj[1] = 0.0;
  unsigned n_active_edges = 0;

  // Make two vectors for storing every reprojection result computed in
  // parallel so that no locking is required in the parallel loop:
  std::vector<float> reprojNorm(n_edges, 0.f);
  std::vector<float> reprojSqNorm(n_edges, 0.f);

  // Use as many threads as there are cores on the CPU:
  tbb::task_scheduler_init init(tbb::task_scheduler_init::automatic);

  for (unsigned e = 0; e < n_edges; ++e) {
    n_active_edges += active_flag[data_counter * n_edges +e];
  }

  tbb::parallel_for(0U, n_active_edges, [&](unsigned e) {
    if ((std::find(bad_associations.begin(), bad_associations.end(), e) == bad_associations.end())) {

      Matrix<float,6,1> cam_eta = Map<Matrix<float,6,1>>(&cam_beliefs_eta_[measurements_camIDs[e] * 6]);
      Matrix<float,6,6> cam_lam = Map<Matrix<float,6,6>>(&cam_beliefs_lambda_[measurements_camIDs[e] * 36]);
      VectorXf cam_mu = cam_lam.transpose().inverse() * cam_eta;

      Vector3f lmk_eta = Map<Vector3f>(&lmk_beliefs_eta_[measurements_lIDs[e] * 3]);
      Matrix3f lmk_lam = Map<Matrix3f>(&lmk_beliefs_lambda_[measurements_lIDs[e] * 9]);
      Vector3f lmk_mu = lmk_lam.transpose().inverse() * lmk_eta;

      // cout << "mus" << cam_mu << lmk_mu << "\n";

      Vector3f w;
      w << cam_mu(3), cam_mu(4), cam_mu(5);
      Matrix3f R = eigenso3exp(w);

      // cout << "R" << R << "\n";

      Vector3f pcf = R * lmk_mu;
      // cout << "pcf" << pcf << "\n";

      pcf(0) += cam_mu(0);
      pcf(1) += cam_mu(1);
      pcf(2) += cam_mu(2);

      Vector3f predicted = (K * pcf) / pcf(2);
      Vector2f residuals = Map<Vector2f>(&measurements_[2*e]);
      // cout << "predicted" << predicted << "\n";

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
    // n_active_edges += active_flag[data_counter * n_edges +e];
    reproj[0] += reprojNorm[e];
    reproj[1] += reprojSqNorm[e];
  }

  cout << "Number of active edges: " << n_active_edges << "\n";
  reproj[0] /= n_active_edges;
}

void eval_reproj(float* reproj, unsigned n_edges, std::vector<unsigned int> active_flag, int data_counter,
                      float* cam_beliefs_eta_, float* cam_beliefs_lambda_, float* lmk_beliefs_eta_, float* lmk_beliefs_lambda_,
                      unsigned* measurements_camIDs, unsigned* measurements_lIDs, float* measurements_, float* K_,
                      const std::vector<unsigned int>& bad_associations=std::vector<unsigned int>() ) 
{
  Matrix3f K = Map<Matrix3f>(K_).transpose();
  reproj[0] = 0.0;
  reproj[1] = 0.0;
  int n_active_edges = 0;
  for (unsigned e = 0; e < n_edges; ++e) {
    if (active_flag[data_counter * n_edges + e] == 1) {
      // If edge doesn't have a bad data association
      if ((std::find(bad_associations.begin(), bad_associations.end(), e) == bad_associations.end())) {

        Matrix<float,6,1> cam_eta = Map<Matrix<float,6,1>>(&cam_beliefs_eta_[measurements_camIDs[e] * 6]);
        Matrix<float,6,6> cam_lam = Map<Matrix<float,6,6>>(&cam_beliefs_lambda_[measurements_camIDs[e] * 36]);
        VectorXf cam_mu = cam_lam.transpose().inverse() * cam_eta;

        Vector3f lmk_eta = Map<Vector3f>(&lmk_beliefs_eta_[measurements_lIDs[e] * 3]);
        Matrix3f lmk_lam = Map<Matrix3f>(&lmk_beliefs_lambda_[measurements_lIDs[e] * 9]);
        Vector3f lmk_mu = lmk_lam.transpose().inverse() * lmk_eta;

        // cout << "mus" << cam_mu << lmk_mu << "\n";

        Vector3f w;
        w << cam_mu(3), cam_mu(4), cam_mu(5);
        Matrix3f R = eigenso3exp(w);

        // cout << "R" << R << "\n";

        Vector3f pcf = R * lmk_mu;
        // cout << "pcf" << pcf << "\n";

        pcf(0) += cam_mu(0);
        pcf(1) += cam_mu(1);
        pcf(2) += cam_mu(2);

        Vector3f predicted = (K * pcf) / pcf(2);
        Vector2f residuals = Map<Vector2f>(&measurements_[2*e]);
        // cout << "predicted" << predicted << "\n";

        residuals(0) -= predicted(0);
        residuals(1) -= predicted(1);

        reproj[0] += residuals.norm();
        reproj[1] += 0.5 * residuals.squaredNorm();
      }
    }


    n_active_edges += active_flag[data_counter * n_edges +e];
  }
  // if !(bad_associations == NULL) {
    n_active_edges -= bad_associations.size();
  // }
  cout << "Number of active edges: " << n_active_edges << "\n";
  reproj[0] /= n_active_edges;
}




// Returns compute set that updates the factors at the new linearisation point which is the current estimate of the beliefs
ComputeSet buildRelineariseCS(Graph &graph, 
                                unsigned n_keyframes,
                                unsigned n_points,
                                unsigned n_edges,
                                unsigned cam_dofs,
                                unsigned lmks_per_tile,
                                unsigned factors_per_tile,
                                unsigned n_cam_tiles,
                                unsigned n_lmk_tiles,
                                std::vector<unsigned> measurements_camIDs, 
                                std::vector<unsigned> measurements_lIDs, 
                                Tensor measurements, 
                                Tensor meas_variances,
                                Tensor K,
                                Tensor cam_beliefs_eta, 
                                Tensor cam_beliefs_lambda,
                                Tensor lmk_beliefs_eta,
                                Tensor lmk_beliefs_lambda,
                                Tensor factor_potentials_eta,
                                Tensor factor_potentials_lambda,
                                Tensor robust_flag) {

  // A compute set is a set of vertices that are executed in parallel.
  ComputeSet cs_relinearise = graph.addComputeSet("cs_relinearise");

  // Add vertices to update keyframe node beliefs.
  for (unsigned edgeID = 0; edgeID < n_edges; ++edgeID) {

    VertexRef vtx = graph.addVertex(cs_relinearise, "RelineariseFactorVertex");

    unsigned adj_cID = measurements_camIDs[edgeID];
    unsigned adj_lID = measurements_lIDs[edgeID];

    // Set tile mapping of vertex.    edge_tile = n_cam_tiles + n_lmk_tiles + floor((float)edgeID / (float)factors_per_tile);

    graph.setTileMapping(vtx, n_cam_tiles + floor((float)edgeID / (float)factors_per_tile));
    // graph.setTileMapping(vtx, n_cam_tiles + floor((float)adj_lID / (float)lmks_per_tile));

    // Connect data to the fields of the vertex. ie. define edges.
    graph.connect(vtx["measurement"], measurements.slice(edgeID * 2, (edgeID + 1) * 2));
    graph.connect(vtx["meas_variance"], meas_variances[edgeID]);
    graph.connect(vtx["robust_flag"], robust_flag[edgeID]);
    graph.connect(vtx["K_"], K.slice(edgeID * 9, (edgeID + 1) * 9));
    graph.connect(vtx["kf_belief_eta_"], cam_beliefs_eta.slice(adj_cID * cam_dofs, (adj_cID + 1) * cam_dofs));
    graph.connect(vtx["kf_belief_lambda_"], cam_beliefs_lambda.slice(adj_cID * cam_dofs * cam_dofs, (adj_cID + 1) * cam_dofs * cam_dofs));
    graph.connect(vtx["lmk_belief_eta_"], lmk_beliefs_eta.slice(adj_lID * 3, (adj_lID + 1) * 3));
    graph.connect(vtx["lmk_belief_lambda_"], lmk_beliefs_lambda.slice(adj_lID * 3 * 3, (adj_lID + 1) * 3 * 3));
    graph.connect(vtx["factor_eta_"], factor_potentials_eta.slice(edgeID * (cam_dofs + 3), (edgeID + 1) * (cam_dofs + 3)));
    graph.connect(vtx["factor_lambda_cc_"], factor_potentials_lambda.slice(edgeID * (cam_dofs + 3) * (cam_dofs + 3), edgeID * (cam_dofs + 3) * (cam_dofs + 3) + cam_dofs * cam_dofs));
    graph.connect(vtx["factor_lambda_ll_"], factor_potentials_lambda.slice((edgeID + 1) * (cam_dofs + 3) * (cam_dofs + 3) - 3 * 3, (edgeID + 1) * (cam_dofs + 3) * (cam_dofs + 3)));
    graph.connect(vtx["factor_lambda_cl_"], factor_potentials_lambda.slice(edgeID * (cam_dofs + 3) * (cam_dofs + 3) + cam_dofs * cam_dofs, edgeID * (cam_dofs + 3) * (cam_dofs + 3) + cam_dofs * cam_dofs + cam_dofs * 3));
    graph.connect(vtx["factor_lambda_lc_"], factor_potentials_lambda.slice(edgeID * (cam_dofs + 3) * (cam_dofs + 3) + cam_dofs * cam_dofs + cam_dofs * 3, (edgeID + 1) * (cam_dofs + 3) * (cam_dofs + 3) - 3 * 3));
  }

  return cs_relinearise;
}


// Builds a program to update the beliefs in 4 series. 
vector<ComputeSet> buildUpdateBeliefsProg(Graph &graph, 
                                unsigned n_keyframes,
                                unsigned n_points,
                                unsigned cam_dofs,
                                unsigned max_nkfedges,
                                unsigned max_nlmkedges,
                                Tensor cam_messages_eta,
                                Tensor cam_messages_lambda,
                                Tensor lmk_messages_eta,
                                Tensor lmk_messages_lambda,
                                Tensor cam_beliefs_eta,
                                Tensor cam_beliefs_lambda,
                                Tensor lmk_beliefs_eta,
                                Tensor lmk_beliefs_lambda) {

  Sequence prog;
  // Reduce along incoming messages
  Operation ADD = Operation::ADD;
  ReduceParams params(ADD);

  OptionFlags reduceOpts {
    {"accumType.interTile", "float"},
    {"accumType.inVertex", "float"}
  };
  vector<ComputeSet> cs;
  reduceWithOutput(graph, cam_messages_eta.reshape({n_keyframes, max_nkfedges + 1, cam_dofs}), 
                    cam_beliefs_eta.reshape({n_keyframes, cam_dofs}), {1}, params, cs, "", reduceOpts);
  reduceWithOutput(graph, cam_messages_lambda.reshape({n_keyframes, max_nkfedges + 1, cam_dofs*cam_dofs}), 
                    cam_beliefs_lambda.reshape({n_keyframes, cam_dofs*cam_dofs}), {1}, params, cs, "", reduceOpts);
  reduceWithOutput(graph, lmk_messages_eta.reshape({n_points, max_nlmkedges + 1, 3}), 
                    lmk_beliefs_eta.reshape({n_points, 3}), {1}, params, cs, "", reduceOpts);
  reduceWithOutput(graph, lmk_messages_lambda.reshape({n_points, max_nlmkedges + 1, 9}), 
                    lmk_beliefs_lambda.reshape({n_points, 9}), {1}, params, cs, "", reduceOpts);

  return cs;
}



// Builds a program to weaken the strength of the priors at each variable node.
ComputeSet buildWeakenPriorCS(Graph &graph, 
                                unsigned n_keyframes,
                                unsigned n_points,
                                unsigned cam_dofs,
                                unsigned max_nkfedges,
                                unsigned max_nlmkedges,
                                unsigned cams_per_tile,
                                unsigned lmks_per_tile,
                                unsigned n_cam_tiles,
                                Tensor cam_messages_eta,
                                Tensor cam_messages_lambda,
                                Tensor lmk_messages_eta,
                                Tensor lmk_messages_lambda,
                                Tensor cam_scaling,
                                Tensor lmk_scaling,
                                Tensor cam_weaken_flag,
                                Tensor lmk_weaken_flag) {

  // A compute set is a set of vertices that are executed in parallel.
  ComputeSet cs_weaken_prior = graph.addComputeSet("cs_weaken_prior");

  for (unsigned cID = 0; cID < n_keyframes; ++cID) {
    VertexRef vx = graph.addVertex(cs_weaken_prior, "WeakenPriorVertex");
    graph.setTileMapping(vx, floor((float)cID / (float)cams_per_tile));

    graph.connect(vx["scaling"], cam_scaling[cID]);
    graph.connect(vx["weaken_flag"], cam_weaken_flag[cID]);
    graph.connect(vx["prior_eta"], cam_messages_eta.slice(cID * cam_dofs * (max_nkfedges + 1), cID * cam_dofs * (max_nkfedges + 1) + cam_dofs));
    graph.connect(vx["prior_lambda"], cam_messages_lambda.slice(cID * cam_dofs * cam_dofs * (max_nkfedges + 1), cID * cam_dofs * cam_dofs * (max_nkfedges + 1) + cam_dofs * cam_dofs));
  }
  for (unsigned lID = 0; lID < n_points; ++lID) {
    VertexRef vx = graph.addVertex(cs_weaken_prior, "WeakenPriorVertex");
    graph.setTileMapping(vx, floor((float)lID / (float)lmks_per_tile) + n_cam_tiles);

    graph.connect(vx["scaling"], lmk_scaling[lID]);
    graph.connect(vx["weaken_flag"], lmk_weaken_flag[lID]);
    graph.connect(vx["prior_eta"], lmk_messages_eta.slice((max_nlmkedges + 1) * 3 * lID, (max_nlmkedges + 1) * 3 * lID + 3 ));
    graph.connect(vx["prior_lambda"], lmk_messages_lambda.slice((max_nlmkedges + 1) * 9 * lID, (max_nlmkedges + 1) * 9 * lID + 9 ));
  }

  return cs_weaken_prior;
}


// This function returns  compute set that computes the outgoing messages at all factor nodes.
vector<ComputeSet> buildComputeMessagesCS(Graph &graph, 
                                unsigned n_keyframes,
                                unsigned n_points,
                                unsigned n_edges,
                                unsigned cam_dofs,
                                unsigned max_nkfedges,
                                unsigned max_nlmkedges,
                                unsigned lmks_per_tile,
                                unsigned factors_per_tile,
                                unsigned n_cam_tiles, 
                                unsigned n_lmk_tiles,
                                std::vector<unsigned> n_edges_per_kf, 
                                std::vector<unsigned> n_edges_per_lmk, 
                                std::vector<unsigned> measurements_camIDs,
                                std::vector<unsigned> measurements_lIDs,
                                Tensor cams_dofs,
                                Tensor lmk_dofs,
                                Tensor active_flag,
                                Tensor factor_potentials_eta,
                                Tensor factor_potentials_lambda,
                                Tensor cam_messages_eta,
                                Tensor cam_messages_lambda,
                                Tensor lmk_messages_eta,
                                Tensor lmk_messages_lambda,
                                Tensor pcam_messages_eta,
                                Tensor pcam_messages_lambda,
                                Tensor plmk_messages_eta,
                                Tensor plmk_messages_lambda,
                                Tensor cam_beliefs_eta,
                                Tensor cam_beliefs_lambda,
                                Tensor lmk_beliefs_eta,
                                Tensor lmk_beliefs_lambda,
                                Tensor measurements,
                                Tensor meas_variances,
                                Tensor K,
                                Tensor damping,
                                Tensor damping_count,
                                Tensor mu,
                                Tensor oldmu,
                                Tensor dmu,
                                Tensor robust_flag) {

  vector<ComputeSet> cs(2);


  cs[0] = graph.addComputeSet("cs_compmess_prep");



  // A compute set is a set of vertices that are executed in parallel.
  // This compute set contains the vertices to compute the outgoing messages at every factor node. 
  cs[1] = graph.addComputeSet("cs_computemessages");

  // Loop through factor nodes in the graph
  for (unsigned edgeID = 0; edgeID < n_edges; ++edgeID) {
    unsigned adj_cID = measurements_camIDs[edgeID];
    unsigned adj_lID = measurements_lIDs[edgeID];

    unsigned tm = n_cam_tiles + floor((float)edgeID / (float)factors_per_tile);
    // unsigned tm = n_cam_tiles + floor((float)adj_lID / (float)lmks_per_tile);

    // unsigned tm = 0;
    // // Choose tile mapping for edge. Strategy is to map half of each nodes edges onto the same tile.
    // if (edgeID % 2 == 0) {
    //   tm = vnID0;
    // }    
    // else {
    //   tm = vnID1;
    // }

    VertexRef vx_prep_mess = graph.addVertex(cs[0], "PrepMessageVertex");

    // Create vertices for the factor node. 2 vertices for computing eta messages in each direction and 2 for lambda messages
    VertexRef vx_cam_eta = graph.addVertex(cs[1], "ComputeCamMessageEtaVertex");  // varnode0 is outedge, varnode1 is nonoutedge
    VertexRef vx_cam_lambda = graph.addVertex(cs[1], "ComputeCamMessageLambdaVertex");

    VertexRef vx_lmk_eta = graph.addVertex(cs[1], "ComputeLmkMessageEtaVertex");  // varnode1 is outedge, varnode0 is nonoutedge
    VertexRef vx_lmk_lambda = graph.addVertex(cs[1], "ComputeLmkMessageLambdaVertex");

    // Set tile mapping of vertex.
    graph.setTileMapping(vx_prep_mess, tm);
    graph.setTileMapping(vx_cam_eta, tm);
    graph.setTileMapping(vx_cam_lambda, tm);
    graph.setTileMapping(vx_lmk_eta, tm);
    graph.setTileMapping(vx_lmk_lambda, tm);

    // Index of factor node at the given variable node. 
    // Want to find how many times the variableID appears in measurements_nodeIDs before the edge we are at now. 
    unsigned edges_c = 0;
    unsigned edges_l = 0;
    for (unsigned i = 0; i < edgeID; ++i) {
      if (measurements_camIDs[i] == adj_cID) {
        edges_c += 1;
      }
    }
    unsigned count_ledges_l = 0;
    for (unsigned i = 0; i < edgeID; ++i) {
      if (measurements_lIDs[i] == adj_lID) {
        edges_l += 1;
      }
    }

    // Connect vertex fields
    graph.connect(vx_prep_mess["damping"], damping[edgeID]);
    graph.connect(vx_prep_mess["damping_count"], damping_count[edgeID]);
    graph.connect(vx_prep_mess["active_flag"], active_flag[edgeID]);
    graph.connect(vx_prep_mess["robust_flag"], robust_flag[edgeID]);
    graph.connect(vx_prep_mess["mu"], mu.slice(edgeID * (cam_dofs + 3), (edgeID + 1) * (cam_dofs + 3) ));
    graph.connect(vx_prep_mess["oldmu"], oldmu.slice(edgeID * (cam_dofs + 3), (edgeID + 1) * (cam_dofs + 3) ));
    graph.connect(vx_prep_mess["dmu"], dmu[edgeID]);
    graph.connect(vx_prep_mess["measurement"], measurements.slice(edgeID * 2, (edgeID + 1) * 2));
    graph.connect(vx_prep_mess["meas_variance"], meas_variances[edgeID]);
    graph.connect(vx_prep_mess["K_"], K.slice(edgeID * 9, (edgeID + 1) * 9));
    graph.connect(vx_prep_mess["kf_belief_eta_"], cam_beliefs_eta.slice(adj_cID * cam_dofs, (adj_cID + 1) * cam_dofs));
    graph.connect(vx_prep_mess["kf_belief_lambda_"], cam_beliefs_lambda.slice(adj_cID * cam_dofs * cam_dofs, (adj_cID + 1) * cam_dofs * cam_dofs));
    graph.connect(vx_prep_mess["lmk_belief_eta_"], lmk_beliefs_eta.slice(adj_lID * 3, (adj_lID + 1) * 3));
    graph.connect(vx_prep_mess["lmk_belief_lambda_"], lmk_beliefs_lambda.slice(adj_lID * 3 * 3, (adj_lID + 1) * 3 * 3));
    graph.connect(vx_prep_mess["factor_eta_"], factor_potentials_eta.slice(edgeID * (cam_dofs + 3), (edgeID + 1) * (cam_dofs + 3)));
    graph.connect(vx_prep_mess["factor_lambda_cc_"], factor_potentials_lambda.slice(edgeID * (cam_dofs + 3) * (cam_dofs + 3), edgeID * (cam_dofs + 3) * (cam_dofs + 3) + cam_dofs * cam_dofs));
    graph.connect(vx_prep_mess["factor_lambda_ll_"], factor_potentials_lambda.slice((edgeID + 1) * (cam_dofs + 3) * (cam_dofs + 3) - 3 * 3, (edgeID + 1) * (cam_dofs + 3) * (cam_dofs + 3)));
    graph.connect(vx_prep_mess["factor_lambda_cl_"], factor_potentials_lambda.slice(edgeID * (cam_dofs + 3) * (cam_dofs + 3) + cam_dofs * cam_dofs, edgeID * (cam_dofs + 3) * (cam_dofs + 3) + cam_dofs * cam_dofs + cam_dofs * 3));
    graph.connect(vx_prep_mess["factor_lambda_lc_"], factor_potentials_lambda.slice(edgeID * (cam_dofs + 3) * (cam_dofs + 3) + cam_dofs * cam_dofs + cam_dofs * 3, (edgeID + 1) * (cam_dofs + 3) * (cam_dofs + 3) - 3 * 3));
 
    // Message is to keyframe node
    graph.connect(vx_cam_eta["damping"], damping[edgeID]);
    graph.connect(vx_cam_eta["active_flag"], active_flag[edgeID]);
    graph.connect(vx_cam_eta["outedge_dofs"], cams_dofs[adj_cID]);
    graph.connect(vx_cam_eta["nonoutedge_dofs"], lmk_dofs[adj_lID]);
    graph.connect(vx_cam_eta["f_outedge_eta_"], factor_potentials_eta.slice(edgeID * (cam_dofs + 3), edgeID * (cam_dofs + 3) + cam_dofs));
    graph.connect(vx_cam_eta["f_nonoutedge_eta_"], factor_potentials_eta.slice(edgeID * (cam_dofs + 3) + cam_dofs, (edgeID + 1) * (cam_dofs + 3)));
    graph.connect(vx_cam_eta["f_noe_noe_lambda_"], factor_potentials_lambda.slice((edgeID + 1) * (cam_dofs + 3) * (cam_dofs + 3) - 3 * 3, (edgeID + 1) * (cam_dofs + 3) * (cam_dofs + 3)));
    graph.connect(vx_cam_eta["f_oe_noe_lambda_"], factor_potentials_lambda.slice(edgeID * (cam_dofs + 3) * (cam_dofs + 3) + cam_dofs * cam_dofs, edgeID * (cam_dofs + 3) * (cam_dofs + 3) + cam_dofs * cam_dofs + cam_dofs * 3));
    graph.connect(vx_cam_eta["belief_nonoutedge_eta_"], lmk_beliefs_eta.slice(adj_lID * 3, (adj_lID + 1) * 3));
    graph.connect(vx_cam_eta["belief_nonoutedge_lambda_"], lmk_beliefs_lambda.slice(adj_lID * 3 * 3, (adj_lID + 1) * 3 * 3));
    graph.connect(vx_cam_eta["pmess_nonoutedge_eta_"], 
      plmk_messages_eta.slice((max_nlmkedges + 1) * 3 * adj_lID + 3 * (edges_l + 1), (max_nlmkedges + 1) * 3 * adj_lID + 3 * (edges_l + 2))); 
    graph.connect(vx_cam_eta["pmess_nonoutedge_lambda_"], 
      plmk_messages_lambda.slice((max_nlmkedges + 1) * 9 * adj_lID + 9 * (edges_l + 1), (max_nlmkedges + 1) * 9 * adj_lID + 9 * (edges_l + 2))); 
    graph.connect(vx_cam_eta["pmess_outedge_eta_"], 
      pcam_messages_eta.slice((max_nkfedges + 1) * cam_dofs * adj_cID + cam_dofs * (edges_c + 1), (max_nkfedges + 1) * cam_dofs * adj_cID + cam_dofs * (edges_c + 2))); 
    graph.connect(vx_cam_eta["mess_outedge_eta_"], 
      cam_messages_eta.slice((max_nkfedges + 1) * cam_dofs * adj_cID + cam_dofs * (edges_c + 1), (max_nkfedges + 1) * cam_dofs * adj_cID + cam_dofs * (edges_c + 2)));  

    graph.connect(vx_cam_lambda["outedge_dofs"], cams_dofs[adj_cID]);
    graph.connect(vx_cam_lambda["nonoutedge_dofs"], lmk_dofs[adj_lID]);
    graph.connect(vx_cam_lambda["active_flag"], active_flag[edgeID]);
    graph.connect(vx_cam_lambda["f_oe_oe_lambda_"], factor_potentials_lambda.slice(edgeID * (cam_dofs + 3) * (cam_dofs + 3), edgeID * (cam_dofs + 3) * (cam_dofs + 3) + cam_dofs * cam_dofs));
    graph.connect(vx_cam_lambda["f_noe_noe_lambda_"], factor_potentials_lambda.slice((edgeID + 1) * (cam_dofs + 3) * (cam_dofs + 3) - 3 * 3, (edgeID + 1) * (cam_dofs + 3) * (cam_dofs + 3)));
    graph.connect(vx_cam_lambda["f_oe_noe_lambda_"], factor_potentials_lambda.slice(edgeID * (cam_dofs + 3) * (cam_dofs + 3) + cam_dofs * cam_dofs, edgeID * (cam_dofs + 3) * (cam_dofs + 3) + cam_dofs * cam_dofs + cam_dofs * 3));
    graph.connect(vx_cam_lambda["f_noe_oe_lambda_"], factor_potentials_lambda.slice(edgeID * (cam_dofs + 3) * (cam_dofs + 3) + cam_dofs * cam_dofs + cam_dofs * 3, (edgeID + 1) * (cam_dofs + 3) * (cam_dofs + 3) - 3 * 3));
    graph.connect(vx_cam_lambda["belief_nonoutedge_lambda_"], lmk_beliefs_lambda.slice(adj_lID * 3 * 3, (adj_lID + 1) * 3 * 3));
    graph.connect(vx_cam_lambda["pmess_nonoutedge_lambda_"], 
      plmk_messages_lambda.slice((max_nlmkedges + 1) * 9 * adj_lID + 9 * (edges_l + 1), (max_nlmkedges + 1) * 9 * adj_lID + 9 * (edges_l + 2))); 
    graph.connect(vx_cam_lambda["mess_outedge_lambda_"], 
      cam_messages_lambda.slice((max_nkfedges + 1) * cam_dofs * cam_dofs * adj_cID + cam_dofs * cam_dofs * (edges_c + 1), (max_nkfedges + 1) * cam_dofs * cam_dofs * adj_cID + cam_dofs * cam_dofs * (edges_c + 2))); 

    // Message is to landmark node
    graph.connect(vx_lmk_eta["damping"], damping[edgeID]);
    graph.connect(vx_lmk_eta["outedge_dofs"], lmk_dofs[adj_lID]);
    graph.connect(vx_lmk_eta["nonoutedge_dofs"], cams_dofs[adj_cID]);
    graph.connect(vx_lmk_eta["active_flag"], active_flag[edgeID]);
    graph.connect(vx_lmk_eta["f_outedge_eta_"], factor_potentials_eta.slice(edgeID * (cam_dofs + 3) + cam_dofs, (edgeID + 1) * (cam_dofs + 3)));
    graph.connect(vx_lmk_eta["f_nonoutedge_eta_"], factor_potentials_eta.slice(edgeID * (cam_dofs + 3), edgeID * (cam_dofs + 3) + cam_dofs));
    graph.connect(vx_lmk_eta["f_noe_noe_lambda_"], factor_potentials_lambda.slice(edgeID * (cam_dofs + 3) * (cam_dofs + 3), edgeID * (cam_dofs + 3) * (cam_dofs + 3) + cam_dofs * cam_dofs));
    graph.connect(vx_lmk_eta["f_oe_noe_lambda_"], factor_potentials_lambda.slice(edgeID * (cam_dofs + 3) * (cam_dofs + 3) + cam_dofs * cam_dofs + cam_dofs * 3, (edgeID + 1) * (cam_dofs + 3) * (cam_dofs + 3) - 3 * 3));
    graph.connect(vx_lmk_eta["belief_nonoutedge_eta_"], cam_beliefs_eta.slice(adj_cID * cam_dofs, (adj_cID + 1) * cam_dofs));
    graph.connect(vx_lmk_eta["belief_nonoutedge_lambda_"], cam_beliefs_lambda.slice(adj_cID * cam_dofs * cam_dofs, (adj_cID + 1) * cam_dofs * cam_dofs));
    graph.connect(vx_lmk_eta["pmess_nonoutedge_eta_"], 
      pcam_messages_eta.slice((max_nkfedges + 1) * cam_dofs * adj_cID + cam_dofs * (edges_c + 1), (max_nkfedges + 1) * cam_dofs * adj_cID + cam_dofs * (edges_c + 2))); 
    graph.connect(vx_lmk_eta["pmess_nonoutedge_lambda_"], 
      pcam_messages_lambda.slice((max_nkfedges + 1) * cam_dofs * cam_dofs * adj_cID + cam_dofs * cam_dofs * (edges_c + 1), (max_nkfedges + 1) * cam_dofs * cam_dofs * adj_cID + cam_dofs * cam_dofs * (edges_c + 2))); 
    graph.connect(vx_lmk_eta["pmess_outedge_eta_"], 
      plmk_messages_eta.slice((max_nlmkedges + 1) * 3 * adj_lID + 3 * (edges_l + 1), (max_nlmkedges + 1) * 3 * adj_lID + 3 * (edges_l + 2))); 
    graph.connect(vx_lmk_eta["mess_outedge_eta_"], 
      lmk_messages_eta.slice((max_nlmkedges + 1) * 3 * adj_lID + 3 * (edges_l + 1), (max_nlmkedges + 1) * 3 * adj_lID + 3 * (edges_l + 2))); 

    graph.connect(vx_lmk_lambda["outedge_dofs"], lmk_dofs[adj_lID]);
    graph.connect(vx_lmk_lambda["nonoutedge_dofs"], cams_dofs[adj_cID]);
    graph.connect(vx_lmk_lambda["active_flag"], active_flag[edgeID]);
    graph.connect(vx_lmk_lambda["f_oe_oe_lambda_"], factor_potentials_lambda.slice((edgeID + 1) * (cam_dofs + 3) * (cam_dofs + 3) - 3 * 3, (edgeID + 1) * (cam_dofs + 3) * (cam_dofs + 3)));
    graph.connect(vx_lmk_lambda["f_noe_noe_lambda_"], factor_potentials_lambda.slice(edgeID * (cam_dofs + 3) * (cam_dofs + 3), edgeID * (cam_dofs + 3) * (cam_dofs + 3) + cam_dofs * cam_dofs));
    graph.connect(vx_lmk_lambda["f_oe_noe_lambda_"], factor_potentials_lambda.slice(edgeID * (cam_dofs + 3) * (cam_dofs + 3) + cam_dofs * cam_dofs + cam_dofs * 3, (edgeID + 1) * (cam_dofs + 3) * (cam_dofs + 3) - 3 * 3));
    graph.connect(vx_lmk_lambda["f_noe_oe_lambda_"], factor_potentials_lambda.slice(edgeID * (cam_dofs + 3) * (cam_dofs + 3) + cam_dofs * cam_dofs, edgeID * (cam_dofs + 3) * (cam_dofs + 3) + cam_dofs * cam_dofs + cam_dofs * 3));
    graph.connect(vx_lmk_lambda["belief_nonoutedge_lambda_"], cam_beliefs_lambda.slice(adj_cID * cam_dofs * cam_dofs, (adj_cID + 1) * cam_dofs * cam_dofs));
    graph.connect(vx_lmk_lambda["pmess_nonoutedge_lambda_"], 
      pcam_messages_lambda.slice((max_nkfedges + 1) * cam_dofs * cam_dofs * adj_cID + cam_dofs * cam_dofs * (edges_c + 1), (max_nkfedges + 1) * cam_dofs * cam_dofs * adj_cID + cam_dofs * cam_dofs * (edges_c + 2))); 
    graph.connect(vx_lmk_lambda["mess_outedge_lambda_"], 
      lmk_messages_lambda.slice((max_nlmkedges + 1) * 9 * adj_lID + 9 * (edges_l + 1), (max_nlmkedges + 1) * 9 * adj_lID + 9 * (edges_l + 2))); 


  }
  return cs;
}


struct Options {
  std::string dir;
  bool inc;
  float transnoise;
  float rotnoise;
  float lmktrans_noise;
  bool verbose;
  bool save_res;
  std::string savedir;
};

Options parseOptions(int argc, char** argv) {
  Options options;
  std::string modeString;

  namespace po = boost::program_options;
  po::options_description desc("Options");
  desc.add_options()
  ("help", "Show command help")
  ("dir",
   po::value<std::string>(&options.dir)->required(),
   "Set the input data directory"
  )
  ("inc",
   po::value<bool>(&options.inc)->default_value(false),
   "SLAM or Bundle adjustment (default BA)"
  )
  ("tn",
   po::value<float>(&options.transnoise)->default_value(0.f),
   "Set keyframe translation noise value"
  )
  ("rn",
   po::value<float>(&options.rotnoise)->default_value(0.f),
   "Set keyframe rotation noise value"
  )
  ("ltn",
   po::value<float>(&options.lmktrans_noise)->default_value(0.f),
   "Set landmark translation noise noise value"
  )
  ("v",
   po::value<bool>(&options.verbose)->default_value(false),
   "Verbose: print beliefs"
  )
  ("save",
   po::value<bool>(&options.save_res)->default_value(false),
   "Save results"
  )
  ("savedir",
   po::value<string>(&options.savedir)->default_value(""),
   "Set the directory to save results to"
  );

  po::variables_map vm;
  po::store(po::parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    std::cout << desc << "\n";
    throw std::runtime_error("help requested");
  }
  po::notify(vm);

  return options;
}


int main(int argc, char** argv) {

  const auto options = parseOptions(argc, argv);

  // Configuration  
  std::string dir = options.dir;
  float cam_trans_noise_std = options.transnoise;
  float cam_rot_noise_std = options.rotnoise;
  cout << "Camera translation noise std to be added: " << cam_trans_noise_std << "\n";
  cout << "Camera rotation noise std to be added: " << cam_rot_noise_std << "\n";
  bool save_res = options.save_res;
  std::string save_dir = options.savedir;
  bool wrong_associations = false;
  bool print_beliefs = options.verbose;
  bool incremental = options.inc; // if true then slam, if false then sfm

  // Load the data describing the properties of the graph. (These variables are not put into the graph but rather used to create it)
  // string dir = "TUM_5_9_0.4_orbd";
  unsigned n_edges = readUnsignedInt(dir + "/n_edges.txt");  // total number of factor nodes / number of edges between variable nodes in the graph.
  unsigned n_keyframes = readUnsignedInt(dir + "/n_keyframes.txt");
  unsigned n_points = readUnsignedInt(dir + "/n_points.txt");
  std::vector<unsigned> pointIDs = readUnsignedIntVector(dir + "/pointIDs.txt");

  // Data about observations required to set up vertices correctly.
  std::vector<unsigned> measurements_camIDs = readUnsignedIntVector(dir + "/measurements_camIDs.txt");
  std::vector<unsigned> measurements_lIDs = readUnsignedIntVector(dir + "/measurements_lIDs.txt");
  std::vector<unsigned> n_edges_per_kf = readUnsignedIntVector(dir + "/n_edges_per_kf.txt");
  std::vector<unsigned> n_edges_per_lmk = readUnsignedIntVector(dir + "/n_edges_per_lmk.txt");

  // Load the data to be fed to the graph through the buffers
  unsigned cam_dofs = readUnsignedInt("/mnt/data/jortiz/data/" + dir + "/cam_dofs.txt");
  std::vector<unsigned> cams_dofs_(n_keyframes, cam_dofs);
  std::vector<unsigned> lmk_dofs_(n_points, 3);

  std::vector<float> cam_priors_mean_ = readFloatVector(dir + "/cam_priors_mean.txt");
  std::vector<float> lmk_priors_mean_ = readFloatVector(dir + "/lmk_priors_mean.txt");

  std::vector<float> cam_priors_eta_ = readFloatVector(dir + "/cam_priors_eta.txt");
  std::vector<float> cam_priors_lambda_ = readFloatVector(dir + "/cam_priors_lambda.txt");
  std::vector<float> lmk_priors_eta_ = readFloatVector(dir + "/lmk_priors_eta.txt");
  std::vector<float> lmk_priors_lambda_ = readFloatVector(dir + "/lmk_priors_lambda.txt");

  std::vector<float> camlog10diffs = readFloatVector(dir + "/camlog10diffs.txt");
  std::vector<float> lmklog10diffs = readFloatVector(dir + "/lmklog10diffs.txt");

  float num_prior_weak_steps = 5.0f;
  float cam_scaling_ [n_keyframes] = {};
  float lmk_scaling_ [n_points] = {};
  for (unsigned cID = 0; cID < n_keyframes; ++cID) {
    cam_scaling_[cID] = pow(10, -camlog10diffs[cID] / num_prior_weak_steps);
  }
  for (unsigned lID = 0; lID < n_points; ++lID) {
    lmk_scaling_[lID] = pow(10, -lmklog10diffs[lID] / num_prior_weak_steps);
  }

  std::vector<float> measurements_ = readFloatVector(dir + "/measurements.txt");
  std::vector<float> meas_variances_ = readFloatVector(dir + "/meas_variances.txt");
  std::vector<float> Ksingle_ = readFloatVector(dir + "/cam_properties.txt");
  float K_ [n_edges * 9] = {};
  for (unsigned i = 0; i < n_edges; ++i) {
    for (unsigned j = 0; j < 9; ++j) {
      K_[i*9 + j] = Ksingle_[j];
    }
  }

  cout << "Loaded data onto host\n";

  std::vector<float> damping_(n_edges, 0.0);
  std::vector<int> damping_count_(n_edges, -15);
  std::vector<float> mu_(n_edges*(cam_dofs + 3), 0.0);
  std::vector<float> oldmu_(n_edges*(cam_dofs + 3), 0.0);
  std::vector<unsigned> robust_flag_(n_edges, 0);



  // // For SLAM load these values
  // cout << "incremental SLAM\n";
  // std::vector<unsigned int> active_flag_ = readUnsignedIntVector(dir + "/active_flag.txt");
  // std::vector<unsigned int> cam_weaken_flag_ = readUnsignedIntVector(dir + "/cam_weaken_flag.txt");
  // std::vector<unsigned int> lmk_weaken_flag_ = readUnsignedIntVector(dir + "/lmk_weaken_flag.txt");

  // For Sfm set these values
  cout << "Structure from motion\n";
  std::vector<unsigned int> active_flag_(n_edges, 1);
  std::vector<unsigned int> cam_weaken_flag_(n_keyframes, 5);
  std::vector<unsigned int> lmk_weaken_flag_(n_points, 5);

  std::vector<unsigned> bad_associations;// = readUnsignedIntVector("/mnt/data/jortiz/data/" + dir + "/bad_associations.txt"); 

  // Create mess floats containing priors to be streamed to IPU
  unsigned max_nkfedges = *max_element(n_edges_per_kf.begin(), n_edges_per_kf.end());
  unsigned max_nlmkedges = *max_element(n_edges_per_lmk.begin(), n_edges_per_lmk.end());

  // Create buffers to hold beliefs when we stream them out
  float cam_beliefs_eta_ [n_keyframes * cam_dofs] = {};
  float cam_beliefs_lambda_ [n_keyframes * cam_dofs * cam_dofs] = {};
  float lmk_beliefs_eta_ [n_points * 3] = {};
  float lmk_beliefs_lambda_ [n_points * 3 * 3] = {};

  // float cam_messages_eta_ [(max_nkfedges + 1) * cam_dofs * n_keyframes] = {};
  float cam_messages_lambda_ [(max_nkfedges + 1) * cam_dofs * cam_dofs * n_keyframes] = {};
  // float pcam_messages_eta_ [(max_nkfedges + 1) * cam_dofs * n_keyframes] = {};
  // float pcam_messages_lambda_ [(max_nkfedges + 1) * cam_dofs * cam_dofs * n_keyframes] = {};

  // float lmk_messages_eta_ [(max_nlmkedges + 1) * 3 * n_points] = {};
  float lmk_messages_lambda_ [(max_nlmkedges + 1) * 9 * n_points] = {};
  // float plmk_messages_eta_ [(max_nlmkedges + 1) * 3 * n_points] = {};
  // float plmk_messages_lambda_ [(max_nlmkedges + 1) * 9 * n_points] = {};



  unsigned cam_active_flag_ [n_keyframes] = {};
  unsigned lmk_active_flag_ [n_points] = {};
  for (unsigned cID = 0; cID < n_keyframes; ++cID) {
    cam_active_flag_[cID] += cam_weaken_flag_[cID];
  }
  for (unsigned lID = 0; lID < n_points; ++lID) {
    lmk_active_flag_[lID] += lmk_weaken_flag_[lID];
  }

  // Set random seed using time
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator (seed);

  if (cam_trans_noise_std != 0.f) {

    // Change initialisation using noise parameters
    std::normal_distribution<float> trans_noise(0.f, cam_trans_noise_std);

    // Add noise to camera position
    for (unsigned cID = 2; cID < n_keyframes; ++cID) {
      for (unsigned i = 0; i < 6; ++i) {
        cam_priors_mean_[cID * 6 + i] += trans_noise(generator);
      }
    }
  }

  if (cam_rot_noise_std != 0.f) {

    // Add noise to camera rotation
    std::normal_distribution<float> rot_noise(0.f, cam_rot_noise_std);

    for (unsigned cID = 2; cID < n_keyframes; ++cID) {

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


  // Change initialisation using noise parameters
  float lmk_trans_noise_std = options.lmktrans_noise;
  std::normal_distribution<float> lmktrans_noise(0.f, lmk_trans_noise_std);

  // Add noise to camera position
  for (unsigned lID = 0; lID < n_points; ++lID) {
    for (unsigned i = 0; i < 3; ++i) {
      lmk_priors_mean_[lID * 3 + i] += lmktrans_noise(generator);
    }
  }


  // float av_depth = 1.f;
  // for (unsigned cID = 0; cID < n_keyframes; ++cID) {
    
  //   Vector3f rot_component;
  //   rot_component << cam_priors_mean_[cID*6 + 3], cam_priors_mean_[cID*6 + 4], cam_priors_mean_[cID*6 + 5];
  //   Matrix3f Rw2c = eigenso3exp(rot_component);
  //   Matrix4f Tw2c;
  //   Tw2c << Rw2c(0,0), Rw2c(0,1), Rw2c(0,2), cam_priors_mean_[cID*6],
  //           Rw2c(1,0), Rw2c(1,1), Rw2c(1,2), cam_priors_mean_[cID*6 + 1],
  //           Rw2c(2,0), Rw2c(2,1), Rw2c(2,2), cam_priors_mean_[cID*6 + 2],
  //           0.0, 0.0, 0.0, 1.0;

  //   Vector4f loc_cam_frame;
  //   loc_cam_frame << 0.0, 0.0, 1.0, 1.0;
  //   Vector4f new_lmk_mu_wf_homog = Tw2c.inverse() * loc_cam_frame;
  //   Vector3f new_lmk_mu_wf;
  //   new_lmk_mu_wf << new_lmk_mu_wf_homog(0), new_lmk_mu_wf_homog(1), new_lmk_mu_wf_homog(2);

  //   for (unsigned e = 0; e < n_edges; ++e) {
  //     if (measurements_camIDs[e] == cID) {
  //       lmk_priors_mean_[measurements_lIDs[e] * 3] = new_lmk_mu_wf(0);
  //       lmk_priors_mean_[measurements_lIDs[e] * 3 + 1] = new_lmk_mu_wf(1);
  //       lmk_priors_mean_[measurements_lIDs[e] * 3 + 2] = new_lmk_mu_wf(2);
  //     }
  //   }

  // }

  // Update beliefs eta with new means
  for (unsigned cID = 0; cID < n_keyframes; ++cID) {
    Matrix<float,6,6> lambda;
    Matrix<float,6,1> mu;
    lambda = Map<Matrix<float,6,6>>(&cam_priors_lambda_[cID * 36]);
    mu = Map<Matrix<float,6,1>>(&cam_priors_mean_[cID*6]);
    // cout << lambda << "\n";
    // cout << mu << "\n";

    Matrix<float,6,1> eta = lambda * mu;
    // cout << eta << "\n";

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
    // cout << eta << "\n";

    lmk_priors_eta_[lID*3] = eta(0);
    lmk_priors_eta_[lID*3 + 1] = eta(1);
    lmk_priors_eta_[lID*3 + 2] = eta(2);
  }
  cout << "Added noise to priors. \n";

  // float fout_eta_ [n_edges * (cam_dofs + 3)] = {};
  // float fout_lambda_ [n_edges * (cam_dofs + 3) * (cam_dofs + 3)] = {};

  std::cout << "Number of keyframe nodes in the graph: " << n_keyframes << '\n';
  std::cout << "Number of landmark nodes in the graph: " << n_points << '\n';
  std::cout << "Number of edges in the graph: " << n_edges << '\n';

  int nchips = 1;
  int counttiles = nchips * 1216 - 1;

  unsigned cams_per_tile = 1;
  unsigned n_cam_tiles = ceil((float)n_keyframes / (float)cams_per_tile);
  unsigned lmks_per_tile = ceil((float)n_points / (float)(counttiles - n_keyframes));
  unsigned n_lmk_tiles = ceil((float)n_points / (float)lmks_per_tile);
  unsigned factors_per_tile = ceil((float)n_edges / (float)(counttiles - n_keyframes));
  unsigned n_factor_tiles = ceil((float)n_edges / (float)factors_per_tile);

  unsigned ntiles = n_cam_tiles + n_factor_tiles;
  // unsigned ntiles = n_cam_tiles + n_lmk_tiles;
  cout << "Number of camera / landmark / factor node per tile: " << cams_per_tile << " / " << lmks_per_tile << " / " << factors_per_tile << "\n";
  cout << "Number of tiles used by camera / landmark / factor nodes and (total) : " << n_cam_tiles << " / " << n_lmk_tiles << " / " << n_factor_tiles << " (" << ntiles << ")\n"; 
  cout << "Number of tiles used by camera / landmark and (total) : " << n_cam_tiles << " / " << n_lmk_tiles << " (" << ntiles << ")\n"; 

  std::cout << "Number of tiles used: " << ntiles << '\n';
  unsigned nIPUs = ceil((float)ntiles / (float)1216);
  if (nIPUs == 0) { nIPUs = 1;}
  if (nIPUs == 3) { nIPUs = 4;}
  if ((4<nIPUs) && (nIPUs<8)) { nIPUs = 8;}
  if ((8<nIPUs) && (nIPUs<16)) { nIPUs = 16;}
  std::cout << "Number of IPUs: " << nIPUs << '\n';
  std::cout << "Number of degrees of freedom at each keyframe node: " << cam_dofs << '\n';

  assert (measurements_camIDs.size() == n_edges);
  assert (measurements_lIDs.size() == n_edges);
  assert (measurements_.size() == 2*n_edges);

  // Compute the factor potentials in advance from the measurements and the measurement sigma
  // hpotentials_eta, hpotentials_lambda = compute_factor_potentials(measurements, sigmas)


  // Create the IPU device
  std::cout << "\nAttaching to IPU device " << std::endl;
  IPUModel ipuModel;
  auto dm = DeviceManager::createDeviceManager();
  auto hwDevices = dm.getDevices(TargetType::IPU, nIPUs);
  poplar::Device device;
  for (auto &testDevice : hwDevices) {
    if (testDevice.attach()) {
      device = std::move(testDevice);
      break;
    }
  }
  std::cout << "\nAttached to IPU device ID " << device.getId() << std::endl;

  //if (device.getId()) {
  //  std::cout << "Could not find a device\n";
  //  exit(-1);
  //}

  // Create the IPU model device
  // IPUModel ipuModel;
  // Device device = ipuModel.createDevice();

  // Create the Graph object 
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  graph.addCodelets("codelets/bp_codelets.cpp");

  // CREATE VARIABLE NODES

  Tensor cams_dofs = graph.addVariable(UNSIGNED_INT, {n_keyframes}, "cams_dofs");
  Tensor lmk_dofs = graph.addVariable(UNSIGNED_INT, {n_points}, "lmk_dofs"); 

  Tensor cam_beliefs_eta = graph.addVariable(FLOAT, {cam_dofs * n_keyframes}, "cam_beliefs_eta");
  Tensor cam_beliefs_lambda = graph.addVariable(FLOAT, {cam_dofs * cam_dofs * n_keyframes}, "cam_beliefs_lambda");

  Tensor lmk_beliefs_eta = graph.addVariable(FLOAT, {3 * n_points}, "lmk_beliefs_eta");
  Tensor lmk_beliefs_lambda = graph.addVariable(FLOAT, {3 * 3 * n_points}, "lmk_beliefs_lambda");

  Tensor cam_scaling = graph.addVariable(FLOAT, {n_keyframes}, "cam_scaling");
  Tensor lmk_scaling = graph.addVariable(FLOAT, {n_points}, "lmk_scaling");
  Tensor cam_weaken_flag = graph.addVariable(UNSIGNED_INT, {n_keyframes}, "cam_weaken_flag");
  Tensor lmk_weaken_flag = graph.addVariable(UNSIGNED_INT, {n_points}, "lmk_weaken_flag");

  // Create Tensor for messages which are stored at the variable node
  Tensor cam_messages_eta = graph.addVariable(FLOAT, {(max_nkfedges + 1) * cam_dofs * n_keyframes}, "cam_messages_eta"); 
  Tensor cam_messages_lambda = graph.addVariable(FLOAT, {(max_nkfedges + 1) * cam_dofs * cam_dofs * n_keyframes}, "cam_messages_lambda");
  Tensor pcam_messages_eta = graph.addVariable(FLOAT, {(max_nkfedges + 1) * cam_dofs * n_keyframes}, "pcam_messages_eta"); 
  Tensor pcam_messages_lambda = graph.addVariable(FLOAT, {(max_nkfedges + 1) * cam_dofs * cam_dofs * n_keyframes}, "pcam_messages_lambda");
  Tensor lmk_messages_eta = graph.addVariable(FLOAT, {(max_nlmkedges + 1) * 3 * n_points}, "lmk_messages_eta"); 
  Tensor lmk_messages_lambda = graph.addVariable(FLOAT, {(max_nlmkedges + 1) * 9 * n_points}, "lmk_messages_lambda");
  Tensor plmk_messages_eta = graph.addVariable(FLOAT, {(max_nlmkedges + 1) * 3 * n_points}, "plmk_messages_eta"); 
  Tensor plmk_messages_lambda = graph.addVariable(FLOAT, {(max_nlmkedges + 1) * 9 * n_points}, "plmk_messages_lambda");

  // Create data streams to write to graph variables
  DataStream instream_cams_dofs = graph.addHostToDeviceFIFO("instream_cams_dofs", UNSIGNED_INT, n_keyframes);
  DataStream instream_lmk_dofs = graph.addHostToDeviceFIFO("instream_lmk_dofs", UNSIGNED_INT, n_points);

  DataStream instream_cam_scaling = graph.addHostToDeviceFIFO("instream_cam_scaling", FLOAT, n_keyframes);
  DataStream instream_lmk_scaling = graph.addHostToDeviceFIFO("instream_lmk_scaling", FLOAT, n_points);
  DataStream instream_cam_weaken_flag = graph.addHostToDeviceFIFO("instream_cam_weaken_flag", UNSIGNED_INT, n_keyframes);
  DataStream instream_lmk_weaken_flag = graph.addHostToDeviceFIFO("instream_lmk_weaken_flag", UNSIGNED_INT, n_points);

  // Create data streams to write into priors vector
  DataStream instream_cam_priors_eta = graph.addHostToDeviceFIFO("instream_cam_priors_eta", FLOAT, cam_dofs * n_keyframes);
  DataStream instream_cam_priors_lambda = graph.addHostToDeviceFIFO("instream_cam_priors_lambda", FLOAT, cam_dofs * cam_dofs * n_keyframes);
  DataStream instream_lmk_priors_eta = graph.addHostToDeviceFIFO("instream_lmk_priors_eta", FLOAT, 3 * n_points);
  DataStream instream_lmk_priors_lambda = graph.addHostToDeviceFIFO("instream_lmk_priors_lambda", FLOAT, 9 * n_points);

  DataStream outstream_cam_priors_eta = graph.addDeviceToHostFIFO("outstream_cam_priors_eta", FLOAT, cam_dofs * n_keyframes);
  DataStream outstream_cam_priors_lambda = graph.addDeviceToHostFIFO("outstream_cam_priors_lambda", FLOAT, cam_dofs * cam_dofs * n_keyframes);
  DataStream outstream_lmk_priors_eta = graph.addDeviceToHostFIFO("outstream_lmk_priors_eta", FLOAT, 3 * n_points);
  DataStream outstream_lmk_priors_lambda = graph.addDeviceToHostFIFO("outstream_lmk_priors_lambda", FLOAT, 9 * n_points);

  // Create data streams to read out beliefs    
  DataStream outstream_cam_beliefs_eta = graph.addDeviceToHostFIFO("outstream_cam_beliefs_eta", FLOAT, n_keyframes * cam_dofs);
  DataStream outstream_cam_beliefs_lambda = graph.addDeviceToHostFIFO("outstream_cam_beliefs_lambda", FLOAT, n_keyframes * cam_dofs * cam_dofs);
  DataStream outstream_lmk_beliefs_eta = graph.addDeviceToHostFIFO("outstream_lmk_beliefs_eta", FLOAT, n_points * 3);
  DataStream outstream_lmk_beliefs_lambda = graph.addDeviceToHostFIFO("outstream_lmk_beliefs_lambda", FLOAT, n_points * 3 * 3);

  // Messages outstream for KL experiment 
  // DataStream outstream_cam_messages_eta = graph.addDeviceToHostFIFO("outstream_cam_messages_eta", FLOAT, (max_nkfedges + 1) * cam_dofs * n_keyframes);
  DataStream outstream_cam_messages_lambda = graph.addDeviceToHostFIFO("outstream_cam_messages_lambda", FLOAT, (max_nkfedges + 1) * cam_dofs * cam_dofs * n_keyframes);
  // DataStream outstream_pcam_messages_eta = graph.addDeviceToHostFIFO("outstream_pcam_messages_eta", FLOAT, (max_nkfedges + 1) * cam_dofs * n_keyframes);
  // DataStream outstream_pcam_messages_lambda = graph.addDeviceToHostFIFO("outstream_pcam_messages_lambda", FLOAT, (max_nkfedges + 1) * cam_dofs * cam_dofs * n_keyframes);

  // DataStream outstream_lmk_messages_eta = graph.addDeviceToHostFIFO("outstream_lmk_messages_eta", FLOAT, (max_nlmkedges + 1) * 3 * n_points);
  DataStream outstream_lmk_messages_lambda = graph.addDeviceToHostFIFO("outstream_lmk_messages_lambda", FLOAT, (max_nlmkedges + 1) * 9 * n_points);
  // DataStream outstream_plmk_messages_eta = graph.addDeviceToHostFIFO("outstream_plmk_messages_eta", FLOAT, (max_nlmkedges + 1) * 3 * n_points);
  // DataStream outstream_plmk_messages_lambda = graph.addDeviceToHostFIFO("outstream_plmk_messages_lambda", FLOAT, (max_nlmkedges + 1) * 9 * n_points);





  // Set tile mapping for keyframe nodes.

  unsigned cam_tile = 0;
  for (unsigned cID = 0; cID < n_keyframes; ++cID) {
    cam_tile = floor((float)cID / (float)cams_per_tile);
    // Set tile mapping for number of edges for a variable node.
    graph.setTileMapping(cams_dofs[cID], cam_tile);
    // Set tile mapping for beliefs
    graph.setTileMapping(cam_beliefs_eta.slice(cID * cam_dofs, (cID + 1) * cam_dofs), cam_tile);
    graph.setTileMapping(cam_beliefs_lambda.slice(cID * cam_dofs * cam_dofs, (cID + 1) * cam_dofs * cam_dofs), cam_tile);

    graph.setTileMapping(cam_scaling[cID], cam_tile);
    graph.setTileMapping(cam_weaken_flag[cID], cam_tile);

    // // Set tile mapping for messages
    // graph.setTileMapping(cam_messages_eta.slice((max_nkfedges + 1) * cam_dofs * cID, (max_nkfedges + 1) * cam_dofs * (cID + 1)), cam_tile);
    // graph.setTileMapping(cam_messages_lambda.slice((max_nkfedges + 1) * cam_dofs * cam_dofs * cID, (max_nkfedges + 1) * cam_dofs * cam_dofs * (cID + 1) ), cam_tile);
    // graph.setTileMapping(pcam_messages_eta.slice((max_nkfedges + 1) * cam_dofs * cID, (max_nkfedges + 1) * cam_dofs * (cID + 1)), cam_tile);
    // graph.setTileMapping(pcam_messages_lambda.slice((max_nkfedges + 1) * cam_dofs * cam_dofs * cID, (max_nkfedges + 1) * cam_dofs * cam_dofs * (cID + 1) ), cam_tile);

    // Store zero padding part of messages on variable node tile
    graph.setTileMapping(cam_messages_eta.slice((max_nkfedges + 1) * cam_dofs * cID + n_edges_per_kf[cID] * cam_dofs, 
                                                (max_nkfedges + 1) * cam_dofs * (cID + 1)), cam_tile);
    graph.setTileMapping(cam_messages_lambda.slice((max_nkfedges + 1) * cam_dofs * cam_dofs * cID + n_edges_per_kf[cID] * cam_dofs * cam_dofs, 
                                                   (max_nkfedges + 1) * cam_dofs * cam_dofs * (cID + 1) ), cam_tile);
    graph.setTileMapping(pcam_messages_eta.slice((max_nkfedges + 1) * cam_dofs * cID + n_edges_per_kf[cID] * cam_dofs, 
                                                 (max_nkfedges + 1) * cam_dofs * (cID + 1)), cam_tile);
    graph.setTileMapping(pcam_messages_lambda.slice((max_nkfedges + 1) * cam_dofs * cam_dofs * cID + n_edges_per_kf[cID] * cam_dofs * cam_dofs, 
                                                    (max_nkfedges + 1) * cam_dofs * cam_dofs * (cID + 1) ), cam_tile);
  }
  // Set tile mapping for landmark nodes.
  unsigned lmk_tile = 0;
  for (unsigned lID = 0; lID < n_points; ++lID) {
    lmk_tile = floor((float)lID / (float)lmks_per_tile);
    // Set tile mapping for number of edges for a variable node.
    graph.setTileMapping(lmk_dofs[lID], lmk_tile + n_cam_tiles);
    // Set tile mapping for beliefs
    graph.setTileMapping(lmk_beliefs_eta.slice(lID * 3, (lID + 1) * 3), lmk_tile + n_cam_tiles);
    graph.setTileMapping(lmk_beliefs_lambda.slice(lID * 3 * 3, (lID + 1) * 3 * 3), lmk_tile + n_cam_tiles);

    graph.setTileMapping(lmk_scaling[lID], lmk_tile + n_cam_tiles);
    graph.setTileMapping(lmk_weaken_flag[lID], lmk_tile + n_cam_tiles);

    // // Set tile mapping for messages
    // graph.setTileMapping(lmk_messages_eta.slice((max_nlmkedges + 1) * 3 * lID, (max_nlmkedges + 1) * 3 * (lID + 1)), lmk_tile + n_cam_tiles);
    // graph.setTileMapping(lmk_messages_lambda.slice((max_nlmkedges + 1) * 9 * lID, (max_nlmkedges + 1) * 9 * (lID + 1)), lmk_tile + n_cam_tiles);
    // graph.setTileMapping(plmk_messages_eta.slice((max_nlmkedges + 1) * 3 * lID, (max_nlmkedges + 1) * 3 * (lID + 1)), lmk_tile + n_cam_tiles);
    // graph.setTileMapping(plmk_messages_lambda.slice((max_nlmkedges + 1) * 9 * lID, (max_nlmkedges + 1) * 9 * (lID + 1)), lmk_tile + n_cam_tiles);

    graph.setTileMapping(lmk_messages_eta.slice((max_nlmkedges + 1) * 3 * lID + n_edges_per_lmk[lID] * 3, (max_nlmkedges + 1) * 3 * (lID + 1)), lmk_tile + n_cam_tiles);
    graph.setTileMapping(lmk_messages_lambda.slice((max_nlmkedges + 1) * 9 * lID + n_edges_per_lmk[lID] * 9, (max_nlmkedges + 1) * 9 * (lID + 1)), lmk_tile + n_cam_tiles);
    graph.setTileMapping(plmk_messages_eta.slice((max_nlmkedges + 1) * 3 * lID + n_edges_per_lmk[lID] * 3, (max_nlmkedges + 1) * 3 * (lID + 1)), lmk_tile + n_cam_tiles);
    graph.setTileMapping(plmk_messages_lambda.slice((max_nlmkedges + 1) * 9 * lID + n_edges_per_lmk[lID] * 9, (max_nlmkedges + 1) * 9 * (lID + 1)), lmk_tile + n_cam_tiles);
  }


  // CREATE FACTOR NODES FOR EACH EDGE

  // Tensors for regulating damping
  Tensor damping = graph.addVariable(FLOAT, {n_edges}, "damping");
  Tensor damping_count = graph.addVariable(INT, {n_edges}, "damping_count");
  Tensor mu = graph.addVariable(FLOAT, {n_edges * (cam_dofs + 3)}, "mu");
  Tensor oldmu = graph.addVariable(FLOAT, {n_edges * (cam_dofs + 3)}, "oldmu");
  Tensor dmu = graph.addVariable(FLOAT, {n_edges}, "dmu");

  Tensor active_flag = graph.addVariable(UNSIGNED_INT, {n_edges}, "active_flag");
  Tensor robust_flag = graph.addVariable(UNSIGNED_INT, {n_edges}, "robust_flag");

  // Measurement information
  Tensor measurements = graph.addVariable(FLOAT, {n_edges * 2}, "measurements");  // 2 is the dimensionality of the measurement at a single measurement node.
  Tensor meas_variances = graph.addVariable(FLOAT, {n_edges}, "meas_variances");
  Tensor K = graph.addVariable(FLOAT, {n_edges * 9}, "K");

  // Factor potentials can be computed from the measurement and measurement variance.
  Tensor factor_potentials_eta = graph.addVariable(FLOAT, {n_edges * (cam_dofs + 3)}, "factor_potentials_eta");
  Tensor factor_potentials_lambda = graph.addVariable(FLOAT, {n_edges * (cam_dofs + 3) * (cam_dofs + 3)}, "factor_potentials_lambda");

  // Create handle to write the measurements and measurement variances at the start. 
  DataStream instream_damping = graph.addHostToDeviceFIFO("instream_damping", FLOAT, n_edges);
  DataStream instream_damping_count = graph.addHostToDeviceFIFO("instream_damping_count", INT, n_edges);
  DataStream outstream_damping = graph.addDeviceToHostFIFO("outstream_damping", FLOAT, n_edges);
  DataStream outstream_damping_count = graph.addDeviceToHostFIFO("outstream_damping_count", INT, n_edges);
  DataStream instream_mu = graph.addHostToDeviceFIFO("instream_mu", FLOAT, n_edges * (cam_dofs + 3));
  DataStream instream_oldmu = graph.addHostToDeviceFIFO("instream_oldmu", FLOAT, n_edges * (cam_dofs + 3));

  DataStream instream_active_flag = graph.addHostToDeviceFIFO("instream_active_flag", UNSIGNED_INT, n_edges);
  DataStream outstream_robust_flag = graph.addDeviceToHostFIFO("outstream_robust_flag", UNSIGNED_INT, n_edges);

  DataStream instream_measurements = graph.addHostToDeviceFIFO("instream_measurements", FLOAT, n_edges * 2);
  DataStream instream_meas_variances = graph.addHostToDeviceFIFO("instream_meas_variances", FLOAT, n_edges);
  DataStream instream_K = graph.addHostToDeviceFIFO("instream_K", FLOAT, n_edges * 9);

  // DataStream outstream_factors_eta = graph.addDeviceToHostFIFO("outstream_factors_eta", FLOAT, n_edges * (cam_dofs + 3));
  // DataStream outstream_factors_lambda = graph.addDeviceToHostFIFO("outstream_factors_lambda", FLOAT, n_edges * (cam_dofs + 3) * (cam_dofs + 3));

  unsigned count_edges_per_kf [n_keyframes] = {};
  unsigned count_edges_per_lmk [n_points] = {};

  unsigned edge_tile = 0;
  for (unsigned edgeID = 0; edgeID < n_edges; ++edgeID) {
    unsigned adj_cID = measurements_camIDs[edgeID];
    unsigned adj_lID = measurements_lIDs[edgeID];

    // unsigned tile = 0;
    // // Choose tile mapping for edge. Strategy is to map half of each nodes edges onto the same tile.
    // if (edgeID % 2 == 0) {
    //   tile = varnodeID0;
    // }    
    // else {
    //   tile = varnodeID1;
    // }
    edge_tile = n_cam_tiles + floor((float)edgeID / (float)factors_per_tile);
    // edge_tile = n_cam_tiles + floor((float)adj_lID / (float)lmks_per_tile);

    // First n_keyframes + n_points tiles are used for variable nodes, so use the next n_edges tiles for factor nodes.
    graph.setTileMapping(damping[edgeID], edge_tile);
    graph.setTileMapping(damping_count[edgeID], edge_tile);
    graph.setTileMapping(mu.slice(edgeID * (cam_dofs + 3), (edgeID + 1) * (cam_dofs + 3)), edge_tile);
    graph.setTileMapping(oldmu.slice(edgeID * (cam_dofs + 3), (edgeID + 1) * (cam_dofs + 3)), edge_tile);
    graph.setTileMapping(dmu[edgeID], edge_tile);

    graph.setTileMapping(active_flag[edgeID], edge_tile);
    graph.setTileMapping(robust_flag[edgeID], edge_tile);

    graph.setTileMapping(measurements.slice(edgeID * 2, (edgeID + 1) * 2), edge_tile);
    graph.setTileMapping(meas_variances[edgeID], edge_tile);
    graph.setTileMapping(K.slice(edgeID * 9, (edgeID + 1) * 9), edge_tile);

    graph.setTileMapping(factor_potentials_eta.slice(edgeID * (cam_dofs + 3), (edgeID + 1) * (cam_dofs + 3)), edge_tile);
    graph.setTileMapping(factor_potentials_lambda.slice(edgeID * (cam_dofs + 3) * (cam_dofs + 3), (edgeID + 1) * (cam_dofs + 3) * (cam_dofs + 3)), edge_tile);

    graph.setTileMapping(cam_messages_eta.slice((max_nkfedges + 1) * cam_dofs * adj_cID + count_edges_per_kf[adj_cID] * cam_dofs, 
                                                (max_nkfedges + 1) * cam_dofs * adj_cID + count_edges_per_kf[adj_cID] * cam_dofs + cam_dofs), edge_tile);
    graph.setTileMapping(cam_messages_lambda.slice((max_nkfedges + 1) * cam_dofs * cam_dofs * adj_cID + count_edges_per_kf[adj_cID] * cam_dofs * cam_dofs, 
                                                   (max_nkfedges + 1) * cam_dofs * cam_dofs * adj_cID + count_edges_per_kf[adj_cID] * cam_dofs * cam_dofs + cam_dofs * cam_dofs), edge_tile);
    graph.setTileMapping(pcam_messages_eta.slice((max_nkfedges + 1) * cam_dofs * adj_cID + count_edges_per_kf[adj_cID] * cam_dofs, 
                                                (max_nkfedges + 1) * cam_dofs * adj_cID + count_edges_per_kf[adj_cID] * cam_dofs + cam_dofs), edge_tile);
    graph.setTileMapping(pcam_messages_lambda.slice((max_nkfedges + 1) * cam_dofs * cam_dofs * adj_cID + count_edges_per_kf[adj_cID] * cam_dofs * cam_dofs, 
                                                   (max_nkfedges + 1) * cam_dofs * cam_dofs * adj_cID + count_edges_per_kf[adj_cID] * cam_dofs * cam_dofs + cam_dofs * cam_dofs), edge_tile);
    graph.setTileMapping(lmk_messages_eta.slice((max_nlmkedges + 1) * 3 * adj_lID + count_edges_per_lmk[adj_lID] * 3, (max_nlmkedges + 1) * 3 * adj_lID + count_edges_per_lmk[adj_lID] * 3 + 3), edge_tile);
    graph.setTileMapping(lmk_messages_lambda.slice((max_nlmkedges + 1) * 9 * adj_lID + count_edges_per_lmk[adj_lID] * 9, (max_nlmkedges + 1) * 9 * adj_lID + count_edges_per_lmk[adj_lID] * 9 + 9), edge_tile);
    graph.setTileMapping(plmk_messages_eta.slice((max_nlmkedges + 1) * 3 * adj_lID + count_edges_per_lmk[adj_lID] * 3, (max_nlmkedges + 1) * 3 * adj_lID + count_edges_per_lmk[adj_lID] * 3 + 3), edge_tile);
    graph.setTileMapping(plmk_messages_lambda.slice((max_nlmkedges + 1) * 9 * adj_lID + count_edges_per_lmk[adj_lID] * 9, (max_nlmkedges + 1) * 9 * adj_lID + count_edges_per_lmk[adj_lID] * 9 + 9), edge_tile);

    count_edges_per_kf[adj_cID] += 1;
    count_edges_per_lmk[adj_lID] += 1;
    }


  auto cs_relinearise = buildRelineariseCS(graph, n_keyframes, n_points, n_edges, cam_dofs, lmks_per_tile, factors_per_tile, n_cam_tiles, n_lmk_tiles,
                                          measurements_camIDs, measurements_lIDs, 
                                          measurements, meas_variances, K, cam_beliefs_eta, cam_beliefs_lambda, lmk_beliefs_eta, lmk_beliefs_lambda,
                                          factor_potentials_eta, factor_potentials_lambda, robust_flag);

  auto cs_ub_red = buildUpdateBeliefsProg(graph, n_keyframes, n_points, cam_dofs, max_nkfedges, max_nlmkedges,
                                          cam_messages_eta, cam_messages_lambda, lmk_messages_eta, lmk_messages_lambda,
                                          cam_beliefs_eta, cam_beliefs_lambda, lmk_beliefs_eta, lmk_beliefs_lambda);

  auto cs_weaken_prior = buildWeakenPriorCS(graph, n_keyframes, n_points, cam_dofs, max_nkfedges, max_nlmkedges,
                                            cams_per_tile, lmks_per_tile, n_cam_tiles,
                                            cam_messages_eta, cam_messages_lambda, lmk_messages_eta, lmk_messages_lambda,
                                            cam_scaling, lmk_scaling, cam_weaken_flag, lmk_weaken_flag);

  auto csvec_computemessages = buildComputeMessagesCS(graph, n_keyframes, n_points, n_edges, cam_dofs, max_nkfedges, max_nlmkedges,
                                                  lmks_per_tile, factors_per_tile, n_cam_tiles, n_lmk_tiles, n_edges_per_kf, n_edges_per_lmk, measurements_camIDs,
                                                  measurements_lIDs, cams_dofs, lmk_dofs, active_flag, factor_potentials_eta, factor_potentials_lambda, 
                                                  cam_messages_eta, cam_messages_lambda, lmk_messages_eta, lmk_messages_lambda,
                                                  pcam_messages_eta, pcam_messages_lambda, plmk_messages_eta, plmk_messages_lambda,
                                                  cam_beliefs_eta, cam_beliefs_lambda, lmk_beliefs_eta, lmk_beliefs_lambda, 
                                                  measurements, meas_variances, K, damping, damping_count, mu, oldmu, dmu, robust_flag);


  Sequence prog_ub;
  // prog_ub.add(PrintTensor("cam beliefs eta before updating", cam_beliefs_eta));
  // prog_ub.add(PrintTensor("cam beliefs eta after updating", cam_beliefs_eta.reshape({n_keyframes, cam_dofs})));
  // prog_ub.add(PrintTensor("cam messages used to update belief", cam_messages_eta.reshape({n_keyframes, max_nkfedges + 1, cam_dofs})[0]));
  // prog_ub.add(PrintTensor("cam messages used to update belief", cam_messages_lambda.reshape({n_keyframes, max_nkfedges + 1, 36})));
  for (const auto &cs : cs_ub_red) { prog_ub.add(Execute(cs));}  // Using reduction
  // prog_ub.add(PrintTensor("cam beliefs eta after updating", cam_beliefs_eta.reshape({n_keyframes, cam_dofs})));
  // prog_ub.add(PrintTensor("cam beliefs lambda after updating", cam_beliefs_lambda.reshape({n_keyframes, 36})));

  Sequence prog_weaken_prior;
  // prog_weaken_prior.add(PrintTensor("cam0 lambda before weakening ", cam_messages_lambda[0]));
  // prog_weaken_prior.add(PrintTensor("lmk0 lambda before weakening ", lmk_messages_lambda[0]));
  prog_weaken_prior.add(Execute(cs_weaken_prior));
  prog_weaken_prior.add(prog_ub);
  // prog_weaken_prior.add(PrintTensor("cam0 lambda after weakening ", cam_messages_lambda[0]));
  // prog_weaken_prior.add(PrintTensor("lmk0 lambda after weakening ", lmk_messages_lambda[0]));

  // Program to write initial data to tiles
  Sequence write_prog;
  write_prog.add(Copy(instream_damping, damping));
  write_prog.add(Copy(instream_damping_count, damping_count));
  write_prog.add(Copy(instream_mu, mu));
  write_prog.add(Copy(instream_oldmu, oldmu));
  write_prog.add(Copy(instream_active_flag, active_flag));
  write_prog.add(Copy(instream_cams_dofs, cams_dofs));
  write_prog.add(Copy(instream_lmk_dofs, lmk_dofs));
  write_prog.add(Copy(instream_cam_scaling, cam_scaling));
  write_prog.add(Copy(instream_lmk_scaling, lmk_scaling));
  write_prog.add(Copy(instream_cam_weaken_flag, cam_weaken_flag));
  write_prog.add(Copy(instream_lmk_weaken_flag, lmk_weaken_flag));
  write_prog.add(Copy(instream_cam_priors_eta, cam_messages_eta.reshape({n_keyframes, max_nkfedges + 1, cam_dofs}).slice({0, 0}, {n_keyframes, 1}) ));
  write_prog.add(Copy(instream_cam_priors_lambda, cam_messages_lambda.reshape({n_keyframes, max_nkfedges + 1, cam_dofs*cam_dofs}).slice({0, 0}, {n_keyframes, 1}) ));
  write_prog.add(Copy(instream_lmk_priors_eta, lmk_messages_eta.reshape({n_points, max_nlmkedges + 1, 3}).slice({0, 0}, {n_points, 1}) ));
  write_prog.add(Copy(instream_lmk_priors_lambda, lmk_messages_lambda.reshape({n_points, max_nlmkedges + 1, 9}).slice({0, 0}, {n_points, 1}) ));
  write_prog.add(Copy(instream_measurements, measurements));
  write_prog.add(Copy(instream_meas_variances, meas_variances));
  write_prog.add(Copy(instream_K, K));

  FloatingPointBehaviour behaviour(true, true, true, false, true);  // inv, div0, oflo, esr, nanoo

  Sequence linearise_prog;
  setFloatingPointBehaviour(graph, linearise_prog, behaviour, "");
  linearise_prog.add(prog_ub);  // To send priors
  linearise_prog.add(Execute(cs_relinearise)); // To compute factors for the first time

  Sequence lprog;
  setFloatingPointBehaviour(graph, linearise_prog, behaviour, "");
  lprog.add(Execute(csvec_computemessages[0]));  // Choose damping factor and relinearise if necessary
  // lprog.add(PrintTensor("\nmeans\n", mu.slice(0,18)));
  // lprog.add(PrintTensor("dmu values\n", dmu.slice(0,100)));
  // lprog.add(PrintTensor("damping values\n", damping.slice(0,100)));
  // lprog.add(PrintTensor("damping count\n", damping_count.slice(0,100)));
  lprog.add(Copy(mu, oldmu));
  lprog.add(Execute(csvec_computemessages[1]));
  lprog.add(prog_ub);

  lprog.add(Copy(cam_messages_eta, pcam_messages_eta)); // Messages from factor node have been used so can be copied to previous message holder
  lprog.add(Copy(cam_messages_lambda, pcam_messages_lambda));
  lprog.add(Copy(lmk_messages_eta, plmk_messages_eta)); 
  lprog.add(Copy(lmk_messages_lambda, plmk_messages_lambda));

  Repeat timing(10, lprog);


  // Program to stream out data from the IPU to the host
  Sequence rprog;
  rprog.add(Copy(cam_beliefs_eta, outstream_cam_beliefs_eta));
  rprog.add(Copy(cam_beliefs_lambda, outstream_cam_beliefs_lambda));
  rprog.add(Copy(lmk_beliefs_eta, outstream_lmk_beliefs_eta));
  rprog.add(Copy(lmk_beliefs_lambda, outstream_lmk_beliefs_lambda));

  rprog.add(Copy(damping, outstream_damping));
  rprog.add(Copy(damping_count, outstream_damping_count));
  rprog.add(Copy(robust_flag, outstream_robust_flag));
  // rprog.add(Copy(cam_messages_eta, outstream_cam_messages_eta));
  // rprog.add(Copy(factor_potentials_eta, outstream_factors_eta));
  // rprog.add(Copy(factor_potentials_lambda, outstream_factors_lambda));

  // Copy messages to host to compute inf flow around graph
  // rprog.add(Copy(cam_messages_eta, outstream_cam_messages_eta));
  rprog.add(Copy(cam_messages_lambda, outstream_cam_messages_lambda));
  // rprog.add(Copy(pcam_messages_eta, outstream_pcam_messages_eta));
  // rprog.add(Copy(pcam_messages_lambda, outstream_pcam_messages_lambda));
  // rprog.add(Copy(lmk_messages_eta, outstream_lmk_messages_eta));
  rprog.add(Copy(lmk_messages_lambda, outstream_lmk_messages_lambda));
  // rprog.add(Copy(plmk_messages_eta, outstream_plmk_messages_eta));
  // rprog.add(Copy(plmk_messages_lambda, outstream_plmk_messages_lambda));

  Sequence read_priors;
  read_priors.add(Copy(cam_messages_eta.reshape({n_keyframes, max_nkfedges + 1, cam_dofs}).slice({0, 0}, {n_keyframes, 1}), outstream_cam_priors_eta));
  read_priors.add(Copy(cam_messages_lambda.reshape({n_keyframes, max_nkfedges + 1, cam_dofs*cam_dofs}).slice({0, 0}, {n_keyframes, 1}), outstream_cam_priors_lambda));
  read_priors.add(Copy(lmk_messages_eta.reshape({n_points, max_nlmkedges + 1, 3}).slice({0, 0}, {n_points, 1}), outstream_lmk_priors_eta));
  read_priors.add(Copy(lmk_messages_lambda.reshape({n_points, max_nlmkedges + 1, 9}).slice({0, 0}, {n_points, 1}), outstream_lmk_priors_lambda));

  Sequence newkf;
  newkf.add(Copy(instream_damping_count, damping_count));
  newkf.add(Copy(instream_cam_priors_eta, cam_messages_eta.reshape({n_keyframes, max_nkfedges + 1, cam_dofs}).slice({0, 0}, {n_keyframes, 1}) ));
  newkf.add(Copy(instream_cam_priors_lambda, cam_messages_lambda.reshape({n_keyframes, max_nkfedges + 1, cam_dofs*cam_dofs}).slice({0, 0}, {n_keyframes, 1}) ));
  newkf.add(Copy(instream_lmk_priors_eta, lmk_messages_eta.reshape({n_points, max_nlmkedges + 1, 3}).slice({0, 0}, {n_points, 1}) ));
  newkf.add(Copy(instream_lmk_priors_lambda, lmk_messages_lambda.reshape({n_points, max_nlmkedges + 1, 9}).slice({0, 0}, {n_points, 1}) ));
  newkf.add(Copy(instream_active_flag, active_flag));
  newkf.add(Copy(instream_cam_weaken_flag, cam_weaken_flag));
  newkf.add(Copy(instream_lmk_weaken_flag, lmk_weaken_flag));
  newkf.add(prog_ub);  // Update belief of new nodes


  // Make control program for timing so we hav only 1 engine run call
  unsigned num_timing_iters = 27;
  Sequence weaken_and_updatetwice;
  weaken_and_updatetwice.add(prog_weaken_prior);
  weaken_and_updatetwice.add(lprog);
  weaken_and_updatetwice.add(lprog);
  Repeat change_prior(5, weaken_and_updatetwice);
  Repeat bulk_iters(num_timing_iters - 11, lprog);
  Sequence timing_prog;
  setFloatingPointBehaviour(graph, timing_prog, behaviour, "");
  timing_prog.add(write_prog);
  timing_prog.add(linearise_prog);
  timing_prog.add(lprog);
  timing_prog.add(change_prior);
  timing_prog.add(bulk_iters);


  // Create the engine
  OptionFlags engineOpts {
    {"debug.instrumentCompute", "true"},
    {"debug.computeInstrumentationLevel", "tile"},
    {"target.workerStackSizeInBytes", "4096"}
  };

  enum ProgamIds {
    WRITE_PROG = 0,
    LINEARISE_PROG,
    LGBP_PROG,
    WEAKEN_PRIORS,
    READ_PROG,
    READ_PRIORS,
    NEW_KEYFRAME,
    TIMING,
    NUM_PROGS
  };
  std::vector<poplar::program::Program> progs =
    {write_prog, linearise_prog, lprog, prog_weaken_prior, rprog, read_priors, newkf, timing};

  Engine engine(graph, progs, engineOpts);
  // Engine engine(graph, {timing_prog}, engineOpts);
  engine.load(device);

  // Connect data streams for streaming in and out data. Connect stream to data on host. 
  engine.connectStream(instream_damping, &damping_[0]);
  engine.connectStream(instream_damping_count, &damping_count_[0]);
  engine.connectStream(instream_mu, &mu_[0]);
  engine.connectStream(instream_oldmu, &oldmu_[0]);

  engine.connectStream(instream_cams_dofs, &cams_dofs_[0]);
  engine.connectStream(instream_lmk_dofs, &lmk_dofs_[0]);

  engine.connectStream(instream_cam_scaling, &cam_scaling_[0]);
  engine.connectStream(instream_lmk_scaling, &lmk_scaling_[0]);
  engine.connectStream(instream_cam_weaken_flag, &cam_weaken_flag_[0], &cam_weaken_flag_[n_keyframes * (n_keyframes -1)]);
  engine.connectStream(instream_lmk_weaken_flag, &lmk_weaken_flag_[0], &lmk_weaken_flag_[n_points * (n_keyframes - 1)]);
  engine.connectStream(instream_active_flag, &active_flag_[0], &active_flag_[n_edges * (n_keyframes - 1)]);

  engine.connectStream(instream_cam_priors_eta, &cam_priors_eta_[0]);
  engine.connectStream(instream_cam_priors_lambda, &cam_priors_lambda_[0]);
  engine.connectStream(instream_lmk_priors_eta, &lmk_priors_eta_[0]);
  engine.connectStream(instream_lmk_priors_lambda, &lmk_priors_lambda_[0]);

  engine.connectStream(instream_measurements, &measurements_[0]);
  engine.connectStream(instream_meas_variances, &meas_variances_[0]);
  engine.connectStream(instream_K, &K_[0]);

  engine.connectStream(outstream_cam_beliefs_eta, cam_beliefs_eta_);
  engine.connectStream(outstream_cam_beliefs_lambda, cam_beliefs_lambda_);
  engine.connectStream(outstream_lmk_beliefs_eta, lmk_beliefs_eta_);
  engine.connectStream(outstream_lmk_beliefs_lambda, lmk_beliefs_lambda_);

  engine.connectStream(outstream_damping, &damping_[0]);
  engine.connectStream(outstream_damping_count, &damping_count_[0]);
  // engine.connectStream(outstream_factors_eta, fout_eta_);
  // engine.connectStream(outstream_factors_lambda, fout_lambda_);

  engine.connectStream(outstream_cam_priors_eta, &cam_priors_eta_[0]);
  engine.connectStream(outstream_cam_priors_lambda, &cam_priors_lambda_[0]);
  engine.connectStream(outstream_lmk_priors_eta, &lmk_priors_eta_[0]);
  engine.connectStream(outstream_lmk_priors_lambda, &lmk_priors_lambda_[0]);

  engine.connectStream(outstream_robust_flag, &robust_flag_[0]);

  // engine.connectStream(outstream_cam_messages_eta, cam_messages_eta_);
  engine.connectStream(outstream_cam_messages_lambda, cam_messages_lambda_);
  // engine.connectStream(outstream_lmk_messages_eta, lmk_messages_eta_);
  engine.connectStream(outstream_lmk_messages_lambda, lmk_messages_lambda_);
  // engine.connectStream(outstream_pcam_messages_eta, pcam_messages_eta_);
  // engine.connectStream(outstream_pcam_messages_lambda, pcam_messages_lambda_);
  // engine.connectStream(outstream_plmk_messages_eta, plmk_messages_eta_);
  // engine.connectStream(outstream_plmk_messages_lambda, plmk_messages_lambda_);


  // Run programs
  Engine::TimerTimePoint time0 = engine.getTimeStamp();
  auto start_time = std::chrono::high_resolution_clock::now();   // Time how long the message passing program takes

  std::cout << "Running program to stream initial data to IPU\n";
  engine.run(WRITE_PROG);
  std::cout << "Initial data streaming complete\n\n";

  std::cout << "Sending priors and computing factor potentials.\n";
  engine.run(LINEARISE_PROG);
  std::cout << "here.\n";

  engine.run(READ_PROG);

  std::cout << "here.\n";


  unsigned data_counter = 0;
  float reproj [2] = {};
  // eval_reproj(reproj, n_edges, active_flag_, data_counter, cam_beliefs_eta_, cam_beliefs_lambda_, lmk_beliefs_eta_, lmk_beliefs_lambda_,
  //                     &measurements_camIDs[0], &measurements_lIDs[0], &measurements_[0], &Ksingle_[0]);
  // cout << "Initial Reprojection error (not parallel): " << reproj[0] << " Cost " << reproj[1] << "\n";
  std::cout << "here.\n";

  eval_reproj_parallel(reproj, n_edges, active_flag_, data_counter, cam_beliefs_eta_, cam_beliefs_lambda_, lmk_beliefs_eta_, lmk_beliefs_lambda_,
                      &measurements_camIDs[0], &measurements_lIDs[0], &measurements_[0], &Ksingle_[0], bad_associations);
  cout << "Initial Reprojection error: " << reproj[0] << " Cost " << reproj[1] << "\n";

  ofstream reproj_file;
  ofstream cost_file;
  ofstream num_relins_file;
  ofstream precision_file;
  ofstream recall_file;
  ofstream cmeans_file;
  ofstream lmeans_file;
  ofstream kl_file;
  if (save_res) {
    // saveBeliefs(dir, 0, n_keyframes, n_points, cam_dofs, cam_beliefs_eta_, cam_beliefs_lambda_, lmk_beliefs_eta_, lmk_beliefs_lambda_);
    cmeans_file.open(save_dir + "/cam_means.txt");
    lmeans_file.open(save_dir + "/lmk_means.txt");
    save_cam_means(cmeans_file, 0, cam_beliefs_eta_, cam_beliefs_lambda_, n_keyframes, cam_active_flag_);  
    save_lmk_means(lmeans_file, 0, lmk_beliefs_eta_, lmk_beliefs_lambda_, n_points, lmk_active_flag_);

    reproj_file.open(save_dir + "/ipu_reproj.txt");
    cost_file.open(save_dir + "/ipu_cost.txt");
    num_relins_file.open(save_dir + "/num_relins.txt");
    reproj_file << reproj[0] << endl;
    cost_file << reproj[1] << endl;
    num_relins_file << 0 << endl; 

    kl_file.open(save_dir + "/kl.txt");
  }
  if (wrong_associations) {
    precision_file.open(save_dir + "/precision.txt");
    recall_file.open(save_dir + "/recall.txt");
  }

  unsigned nrobust_edges = 0;
  unsigned true_pos = 0;
  unsigned false_pos = 0;
  unsigned true_neg = 0;
  unsigned false_neg = 0;
  float precision = 0.0;
  float recall = 0.0;
  float f1 = 0.0;

  int num_relins = 0;
  unsigned iter = 0;
  unsigned niters;
  unsigned n_iters_per_newkf = 700;
  if (incremental) {
    niters = (n_keyframes - 1) * n_iters_per_newkf - 1;
  }
  else {
    niters = 6000;
  }

  for (unsigned i = 0; i < (niters); ++i) {
    std::cout << "\nIteration: " << i << '\n';

    if (incremental) {
      if ((i+1) % n_iters_per_newkf == 0) {
        cout << "********************** Adding new keyframe **************************\n";
        iter = 0;
        data_counter += 1;

          for (unsigned cID = 0; cID < n_keyframes; ++cID) {
            cam_active_flag_[cID] += cam_weaken_flag_[data_counter * n_keyframes + cID];
          }
          for (unsigned lID = 0; lID < n_points; ++lID) {
            lmk_active_flag_[lID] += lmk_weaken_flag_[data_counter * n_points + lID];
          }

        // Use previous kf for prior on next keyframe
        engine.run(READ_PRIORS); // Read current priors back to host
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

        for (unsigned i = 0; i < n_edges; ++i) {
          damping_count_[i] = -15;
        }

        // cout << "new keyframe is initialised with mean: " << previous_kf_mu << "\n";
        // cout << "new landmarks are initialised with mean: " << new_lmk_mu_wf << "\n";

        engine.run(NEW_KEYFRAME);
        engine.run(READ_PROG);
        if (save_res) {
          // Overwrite beliefs saved at previous iteration to include new keyframe and new landmarks
          cout << "Beliefs with new keyframe just initialised saved at iter " << i << "\n";
          // saveBeliefs(dir, i, n_keyframes, n_points, cam_dofs, cam_beliefs_eta_, cam_beliefs_lambda_, lmk_beliefs_eta_, lmk_beliefs_lambda_);
            save_cam_means(cmeans_file, -1, cam_beliefs_eta_, cam_beliefs_lambda_, n_keyframes, cam_active_flag_);  
            save_lmk_means(lmeans_file, -1, lmk_beliefs_eta_, lmk_beliefs_lambda_, n_points, lmk_active_flag_);
        }
      }
    }


    if (((iter+ 1) % 2 == 0) && (iter < num_prior_weak_steps * 2)) {
      cout << "Weakening priors \n";
      engine.run(WEAKEN_PRIORS);
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    engine.run(LGBP_PROG); // synchronous update
    auto t2 = std::chrono::high_resolution_clock::now();
    engine.run(READ_PROG); // Write beliefs out
    auto t3 = std::chrono::high_resolution_clock::now();

    nrobust_edges = 0;
    true_pos = 0;
    false_pos = 0;
    true_neg = 0;
    false_neg = 0;
    for (unsigned i = 0; i < n_edges; ++i) {
      nrobust_edges += robust_flag_[i];

      if (wrong_associations) {
        if (std::find(bad_associations.begin(), bad_associations.end(), i) != bad_associations.end()) {
          /* Edge has bad data association */
          if (robust_flag_[i] == 1) {
            true_pos += 1;
          }
          if (robust_flag_[i] == 0) {
            false_neg += 1;
          }
        }
        else {
          /* Edge has correct data association */
          if (robust_flag_[i] == 1) {
            false_pos += 1;
          }
          else if (robust_flag_[i] == 0) {
            true_neg += 1;
          }
        }
      }
    }
    if (wrong_associations) {
      precision = (float)true_pos / ((float)true_pos + (float)false_pos);
      recall = (float)true_pos / ((float)true_pos + (float)false_neg);
      f1 = 2 * (precision * recall) / (precision + recall);

      cout << "TP " << true_pos << ", FP " << false_pos << " FN " << false_neg << " TN " << true_neg << "\n";
      cout << "Precision " << precision << ", Recall " << recall << "\n";

      precision_file << precision << endl;
      recall_file << recall << endl;
    }

    auto t4 = std::chrono::high_resolution_clock::now();
    // cout << "\n damping: \n";
    // for (unsigned i = 0; i < 20; ++i) {
    //   cout << " " << damping_[i] << " ";
    // }
    // cout << "\n";
    // cout << "\n damping count: \n";
    num_relins = 0;
    for (unsigned i = 0; i < n_edges; ++i) {
      // cout << " " << damping_count_[i] << " ";
      if (damping_count_[i] == -8) {
        num_relins += 1;
      }
    }

    eval_reproj_parallel(reproj, n_edges, active_flag_, data_counter, 
                cam_beliefs_eta_, cam_beliefs_lambda_, lmk_beliefs_eta_, lmk_beliefs_lambda_,
                &measurements_camIDs[0], &measurements_lIDs[0], &measurements_[0], &Ksingle_[0], bad_associations);
    cout << "Reprojection error " << reproj[0] << " // Cost " << reproj[1] << " // Number of relinearisations: " << num_relins;
    cout << " // Number of robust edges " << nrobust_edges << "\n";
    auto t5 = std::chrono::high_resolution_clock::now();

    if (save_res) {
      reproj_file << reproj[0] << endl;
      cost_file << reproj[1] << endl;
      num_relins_file << num_relins << endl;
      // saveBeliefs(dir, i+1, n_keyframes, n_points, cam_dofs, cam_beliefs_eta_, cam_beliefs_lambda_, lmk_beliefs_eta_, lmk_beliefs_lambda_);
      save_cam_means(cmeans_file, i+1, cam_beliefs_eta_, cam_beliefs_lambda_, n_keyframes, cam_active_flag_);  
      save_lmk_means(lmeans_file, i+1, lmk_beliefs_eta_, lmk_beliefs_lambda_, n_points, lmk_active_flag_);
    

      save_message_KL(kl_file, i+1, n_edges, active_flag_, &measurements_camIDs[0], &measurements_lIDs[0], data_counter, max_nkfedges, max_nlmkedges, cam_dofs,
                      cam_messages_lambda_, lmk_messages_lambda_);
                      // cam_messages_eta_, cam_messages_lambda_, lmk_messages_eta_, lmk_messages_lambda_,
                      // pcam_messages_eta_, pcam_messages_lambda_, plmk_messages_eta_, plmk_messages_lambda_);
    }

    if (print_beliefs) {
      // Print beliefs that are streamed out 
      std::cout << "\nKeyframe Eta beliefs: \n";
      for (unsigned i = 0; i <  6; ++i) {
        printf("%.12f  ", cam_beliefs_eta_[6+i]);  
      };
      std::cout << "\nKeyframe Lambda beliefs: \n";
      for (unsigned i = 0; i <  36; ++i) {
        printf("%.12f  ", cam_beliefs_lambda_[36 + i]);  
      }
      std::cout << '\n';

      std::cout << "\nLandmark Eta beliefs: \n";
      for (unsigned i = 0; i <  12; ++i) {
        printf("%.12f  ", lmk_beliefs_eta_[i]);  
      }
      std::cout << "\nLandmark Lambda beliefs: \n";
      for (unsigned i = 0; i <  3 * 3 * 2; ++i) {
        printf("%.12f  ", lmk_beliefs_lambda_[i]);  
      }
      std::cout << '\n';
    }

    // auto t6 = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> lgbpProgTime = t2 - t1;
    // std::chrono::duration<double, std::milli> readProgTime = t3 - t2;
    // std::chrono::duration<double, std::milli> miscTime = t4 - t3;
    // std::chrono::duration<double, std::milli> reprojEvalTime = t5 - t4;
    // std::chrono::duration<double, std::milli> fileIOTime = t6 - t5;
    // std::cout << "Timings (millisecs) LGBP_PROG READ_PROG MISC REPROJ_EVAL FILE_IO: "
    //   << lgbpProgTime.count() << " "
    //   << readProgTime.count() << " "
    //   << miscTime.count() << " "
    //   << reprojEvalTime.count() << " "
    //   << fileIOTime.count() << " "
    //   << "\n";

    iter += 1;
  }

  if (save_res) {
    reproj_file.close();    
    cost_file.close();
    num_relins_file.close();   
    cmeans_file.close();
    lmeans_file.close();
    kl_file.close();
  }
  precision_file.close();
  recall_file.close();


  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> elapsed = end_time - start_time;
  std::cout << "\nElapsed Time  on chrono for " << niters << " synchronous updates: " << elapsed.count() << " micro seconds" << "\n";

  Engine::TimerTimePoint time1 = engine.getTimeStamp();
  string timing_report = engine.reportTiming(time0, time1);
  std::cout << timing_report << "\n";

  engine.resetExecutionProfile();

  // Timing program
  // Engine::TimerTimePoint time0 = engine.getTimeStamp();
  // auto start_time = std::chrono::high_resolution_clock::now();   // Time how long the message passing program takes
  // std::cout << "Executing timing program\n";
  engine.run(TIMING);
  // auto end_time = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::micro> elapsed = end_time - start_time;
  // std::cout << "\nElapsed Time for timing program: " << elapsed.count() << " micro seconds" << "\n";
  // Engine::TimerTimePoint time1 = engine.getTimeStamp();
  // string timing_report = engine.reportTiming(time0, time1);
  // std::cout << timing_report << "\n";
  // engine.run(LINEARISE_PROG);



  // Profiling
  char* log_dir = std::getenv("GC_PROFILE_LOG_DIR");
  std::string direc = log_dir ? std::string(log_dir) : ".";

  // Save intervals report
  std::ofstream fout(direc + "/intervals.csv");
  engine.reportIntervals(fout);

  // Graph Report
  poplar::ProfileValue graphProfile = engine.getGraphProfile();
  std::ofstream graphReport;
  graphReport.open(direc + "/graph.json");
  poplar::serializeToJSON(graphReport, graphProfile);
  graphReport.close();

  // Execution Report
  poplar::ProfileValue execProfile = engine.getExecutionProfile();
  std::ofstream execReport;
  execReport.open(direc + "/execution.json");
  poplar::serializeToJSON(execReport, execProfile);
  execReport.close();

  // const auto &cycles = execProfile["computeSetCyclesByTile"].asVector();
  // int cscId = cs_ub_red.getId();
  // auto numtiles = device.getTarget().getNumTiles();
  // for (auto t=0;t<numtiles;++t) {
  //   auto camtileCycles = cycles[cscId][t].asInt();
  //   cout << t << " tile cycle " << camtileCycles << "\n";
  // }


  return 0;
}
