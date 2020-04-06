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


#include <chrono>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <math.h>

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
  else std::cout << "Unable to open file"; 

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
  else std::cout << "Unable to open file"; 

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


void eval_reproj(float* reproj, unsigned n_edges, std::vector<unsigned int> active_flag, int data_counter, 
                      float* cam_beliefs_eta_, float* cam_beliefs_lambda_, float* lmk_beliefs_eta_, float* lmk_beliefs_lambda_,
                      unsigned* measurements_camIDs, unsigned* measurements_lIDs, float* measurements_, float* K_) 
{
  Matrix3f K = Map<Matrix3f>(K_).transpose();
  reproj[0] = 0.0;
  reproj[1] = 0.0;
  int n_active_edges = 0;
  for (unsigned e = 0; e < n_edges; ++e) {
    if (active_flag[data_counter * n_edges + e] == 1) {
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


    n_active_edges += active_flag[data_counter * n_edges +e];
  }
  cout << "Number of active edges: " << n_active_edges << "\n";
  reproj[0] /= n_active_edges;
}



// Returns compute set that updates the factors at the new linearisation point which is the current estimate of the beliefs
ComputeSet buildRelineariseCS(Graph &graph, 
                                unsigned n_keyframes,
                                unsigned n_points,
                                unsigned n_edges,
                                unsigned cam_dofs,
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
                                Tensor factor_potentials_lambda) {

  // A compute set is a set of vertices that are executed in parallel.
  ComputeSet cs_relinearise = graph.addComputeSet("cs_relinearise");

  // Add vertices to update keyframe node beliefs.
  for (unsigned edgeID = 0; edgeID < n_edges; ++edgeID) {

    VertexRef vtx = graph.addVertex(cs_relinearise, "RelineariseFactorVertex");

    // Set tile mapping of vertex.
    graph.setTileMapping(vtx, n_keyframes + n_points + edgeID);

    unsigned adj_cID = measurements_camIDs[edgeID];
    unsigned adj_lID = measurements_lIDs[edgeID];


    // Connect data to the fields of the vertex. ie. define edges.
    graph.connect(vtx["measurement"], measurements.slice(edgeID * 2, (edgeID + 1) * 2));
    graph.connect(vtx["meas_variance"], meas_variances[edgeID]);
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
  vector<ComputeSet> cs(1);
  cs[0] = graph.addComputeSet("updateBeliefsComputeSet");
  cs[1] = graph.addComputeSet("updateBeliefsComputeSet");
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




// This function returns  compute set that updates the tensors beliefs_eta and beliefs_lambda
// Arguments that are tensors are used as vertex input or output fields
ComputeSet buildUpdateBeliefsCS(Graph &graph, 
                                unsigned n_keyframes,
                                unsigned n_points,
                                unsigned cam_dofs,
                                Tensor cams_dofs,
                                Tensor lmk_dofs,
                                unsigned max_nkfedges,
                                unsigned max_nlmkedges,
                                std::vector<unsigned> n_edges_per_kf, 
                                std::vector<unsigned> n_edges_per_lmk, 
                                Tensor cam_messages_eta,
                                Tensor cam_messages_lambda,
                                Tensor lmk_messages_eta,
                                Tensor lmk_messages_lambda,
                                Tensor cam_beliefs_eta,
                                Tensor cam_beliefs_lambda,
                                Tensor lmk_beliefs_eta,
                                Tensor lmk_beliefs_lambda) {

  // A compute set is a set of vertices that are executed in parallel.
  ComputeSet cs_updatebeliefs = graph.addComputeSet("cs_updatebeliefs");

  // Add vertices to update keyframe node beliefs.
  for (unsigned cID = 0; cID < n_keyframes; ++cID) {
    VertexRef vtx_eta = graph.addVertex(cs_updatebeliefs, "UpdateBeliefEtaVertex");
    VertexRef vtx_lambda = graph.addVertex(cs_updatebeliefs, "UpdateBeliefLambdaVertex");

    // Set tile mapping of vertex.
    graph.setTileMapping(vtx_eta, cID);
    graph.setTileMapping(vtx_lambda, cID);

    // Connect data to the fields of the vertex. ie. define edges.
    graph.connect(vtx_eta["dofs"], cams_dofs[cID]);
    graph.connect(vtx_lambda["dofs"], cams_dofs[cID]);
    graph.connect(vtx_eta["prior_eta"], cam_messages_eta.slice(cID * cam_dofs * (max_nkfedges + 1), cID * cam_dofs * (max_nkfedges + 1) + cam_dofs));
    graph.connect(vtx_lambda["prior_lambda"], cam_messages_lambda.slice(cID * cam_dofs * cam_dofs * (max_nkfedges + 1), cID * cam_dofs * cam_dofs * (max_nkfedges + 1) + cam_dofs * cam_dofs));

    graph.connect(vtx_eta["incoming_messages_eta"], 
      cam_messages_eta.slice(cam_dofs * (max_nkfedges + 1) * cID + cam_dofs , cam_dofs * (max_nkfedges + 1) * (cID + 1)  ));
    graph.connect(vtx_lambda["incoming_messages_lambda"], 
      cam_messages_lambda.slice(cam_dofs * cam_dofs * (max_nkfedges + 1) * cID + cam_dofs * cam_dofs, cam_dofs * cam_dofs * (max_nkfedges + 1) * (cID + 1) ));

    // Output fields of vertex
    graph.connect(vtx_eta["belief_eta"], 
      cam_beliefs_eta.slice(cID * cam_dofs, (cID + 1) * cam_dofs));
    graph.connect(vtx_lambda["belief_lambda"], 
      cam_beliefs_lambda.slice(cID * cam_dofs * cam_dofs, (cID + 1) * cam_dofs * cam_dofs));
  }

  // Add vertices to update landmark node beliefs. 
  for (unsigned lID = 0; lID < n_points; ++lID) {
    VertexRef vtx_eta = graph.addVertex(cs_updatebeliefs, "UpdateBeliefEtaVertex");
    VertexRef vtx_lambda = graph.addVertex(cs_updatebeliefs, "UpdateBeliefLambdaVertex");

    // Set tile mapping of vertex.
    graph.setTileMapping(vtx_eta, lID + n_keyframes);
    graph.setTileMapping(vtx_lambda, lID + n_keyframes);

    // Connect data to the fields of the vertex. ie. define edges.
    graph.connect(vtx_eta["dofs"], lmk_dofs[lID]);
    graph.connect(vtx_lambda["dofs"], lmk_dofs[lID]);
    graph.connect(vtx_eta["prior_eta"], lmk_messages_eta.slice((max_nlmkedges + 1) * 3 * lID, (max_nlmkedges + 1) * 3 * lID + 3 ));
    graph.connect(vtx_lambda["prior_lambda"], lmk_messages_lambda.slice((max_nlmkedges + 1) * 9 * lID, (max_nlmkedges + 1) * 9 * lID + 9 ));

    // std::cout << (max_nlmkedges + 1) * 3 * lID + 3 << ' ' << (max_nlmkedges + 1) * 3 * (lID + 1) << '\n';
    // std::cout << max_nlmkedges << 'max_nlmkedges\n';
    // std::cout << (max_nlmkedges + 1) * 3 << 'max_nlmkedges\n';


    graph.connect(vtx_eta["incoming_messages_eta"], 
      lmk_messages_eta.slice((max_nlmkedges + 1) * 3 * lID + 3, (max_nlmkedges + 1) * 3 * (lID + 1) ));
    graph.connect(vtx_lambda["incoming_messages_lambda"], 
      lmk_messages_lambda.slice((max_nlmkedges + 1) * 9 * lID + 9, (max_nlmkedges + 1) * 9 * (lID + 1) ));

    // Output fields of vertex
    graph.connect(vtx_eta["belief_eta"], 
      lmk_beliefs_eta.slice(lID * 3, (lID + 1) * 3));
    graph.connect(vtx_lambda["belief_lambda"], 
      lmk_beliefs_lambda.slice(lID * 3 * 3, (lID + 1) * 3 * 3));
  }


  return cs_updatebeliefs;
}

// Builds a program to weaken the strength of the priors at each variable node.
ComputeSet buildWeakenPriorCS(Graph &graph, 
                                unsigned n_keyframes,
                                unsigned n_points,
                                unsigned cam_dofs,
                                unsigned max_nkfedges,
                                unsigned max_nlmkedges,
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
    graph.setTileMapping(vx, cID);

    graph.connect(vx["scaling"], cam_scaling[cID]);
    graph.connect(vx["weaken_flag"], cam_weaken_flag[cID]);
    graph.connect(vx["prior_eta"], cam_messages_eta.slice(cID * cam_dofs * (max_nkfedges + 1), cID * cam_dofs * (max_nkfedges + 1) + cam_dofs));
    graph.connect(vx["prior_lambda"], cam_messages_lambda.slice(cID * cam_dofs * cam_dofs * (max_nkfedges + 1), cID * cam_dofs * cam_dofs * (max_nkfedges + 1) + cam_dofs * cam_dofs));
  }
  for (unsigned lID = 0; lID < n_points; ++lID) {
    VertexRef vx = graph.addVertex(cs_weaken_prior, "WeakenPriorVertex");
    graph.setTileMapping(vx, lID + n_keyframes);

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
                                Tensor dmu) {

  vector<ComputeSet> cs(2);


  cs[0] = graph.addComputeSet("cs_compmess_prep");



  // A compute set is a set of vertices that are executed in parallel.
  // This compute set contains the vertices to compute the outgoing messages at every factor node. 
  cs[1] = graph.addComputeSet("cs_computemessages");

  // Loop through factor nodes in the graph
  for (unsigned edgeID = 0; edgeID < n_edges; ++edgeID) {

    unsigned adj_cID = measurements_camIDs[edgeID];
    unsigned adj_lID = measurements_lIDs[edgeID];

    // unsigned tm = 0;
    // // Choose tile mapping for edge. Strategy is to map half of each nodes edges onto the same tile.
    // if (edgeID % 2 == 0) {
    //   tm = vnID0;
    // }    
    // else {
    //   tm = vnID1;
    // }
    unsigned tm = n_keyframes + n_points + edgeID;

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
    unsigned count_ledges_c = 0;
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



int main(int argc, char** argv) {

  if (argc != 2) {
    std::cerr << "Must specify data directory!\n";
    return 1;
  }
  string dir = argv[1];

  // Load the data describing the properties of the graph. (These variables are not put into the graph but rather used to create it)
  // string dir = "TUM_5_9_0.4_orbd";
  unsigned n_edges = readUnsignedInt("../data/" + dir + "/n_edges.txt");  // total number of factor nodes / number of edges between variable nodes in the graph.
  unsigned n_keyframes = readUnsignedInt("../data/" + dir + "/n_keyframes.txt");
  unsigned n_points = readUnsignedInt("../data/" + dir + "/n_points.txt");
  std::vector<unsigned> pointIDs = readUnsignedIntVector("../data/" + dir + "/pointIDs.txt");

  // Data about observations required to set up vertices correctly.
  std::vector<unsigned> measurements_camIDs = readUnsignedIntVector("../data/" + dir + "/measurements_camIDs.txt");
  std::vector<unsigned> measurements_lIDs = readUnsignedIntVector("../data/" + dir + "/measurements_lIDs.txt");
  std::vector<unsigned> n_edges_per_kf = readUnsignedIntVector("../data/" + dir + "/n_edges_per_kf.txt");
  std::vector<unsigned> n_edges_per_lmk = readUnsignedIntVector("../data/" + dir + "/n_edges_per_lmk.txt");

  // Load the data to be fed to the graph through the buffers
  unsigned cam_dofs = readUnsignedInt("../data/" + dir + "/cam_dofs.txt");
  std::vector<unsigned> cams_dofs_(n_keyframes, cam_dofs);
  std::vector<unsigned> lmk_dofs_(n_points, 3);

  std::vector<float> cam_priors_eta_ = readFloatVector("../data/" + dir + "/cam_priors_eta.txt");
  std::vector<float> cam_priors_lambda_ = readFloatVector("../data/" + dir + "/cam_priors_lambda.txt");
  std::vector<float> lmk_priors_eta_ = readFloatVector("../data/" + dir + "/lmk_priors_eta.txt");
  std::vector<float> lmk_priors_lambda_ = readFloatVector("../data/" + dir + "/lmk_priors_lambda.txt");

  std::vector<float> camlog10diffs = readFloatVector("../data/" + dir + "/camlog10diffs.txt");
  std::vector<float> lmklog10diffs = readFloatVector("../data/" + dir + "/lmklog10diffs.txt");
  float num_prior_weak_steps = 5.0f;
  float cam_scaling_ [n_keyframes] = {};
  float lmk_scaling_ [n_points] = {};
  for (unsigned cID = 0; cID < n_keyframes; ++cID) {
    cam_scaling_[cID] = pow(10, -camlog10diffs[cID] / num_prior_weak_steps);
  }
  for (unsigned lID = 0; lID < n_points; ++lID) {
    lmk_scaling_[lID] = pow(10, -lmklog10diffs[lID] / num_prior_weak_steps);
  }

  std::vector<float> measurements_ = readFloatVector("../data/" + dir + "/measurements.txt");
  std::vector<float> meas_variances_ = readFloatVector("../data/" + dir + "/meas_variances.txt");
  std::vector<float> Ksingle_ = readFloatVector("../data/" + dir + "/cam_properties.txt");
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


  // For Sfm set these values
  // std::vector<unsigned int> active_flag_(n_edges, 1);
  // std::vector<unsigned int> cam_weaken_flag_(n_keyframes, 5);
  // std::vector<unsigned int> lmk_weaken_flag_(n_points, 5);
  // For SLAM load these values
  std::vector<unsigned> active_flag_ = readUnsignedIntVector("../data/" + dir + "/active_flag.txt");
  std::vector<unsigned> cam_weaken_flag_ = readUnsignedIntVector("../data/" + dir + "/cam_weaken_flag.txt");
  std::vector<unsigned> lmk_weaken_flag_ = readUnsignedIntVector("../data/" + dir + "/lmk_weaken_flag.txt");

  // Create mess floats containing priors to be streamed to IPU
  unsigned max_nkfedges = *max_element(n_edges_per_kf.begin(), n_edges_per_kf.end());
  unsigned max_nlmkedges = *max_element(n_edges_per_lmk.begin(), n_edges_per_lmk.end());

  float cam_messages_eta_ [(max_nkfedges + 1) * cam_dofs * n_keyframes] = {};  // + 1 for prior
  float cam_messages_lambda_ [(max_nkfedges + 1) * cam_dofs * cam_dofs * n_keyframes] = {};
  float lmk_messages_eta_ [(max_nlmkedges + 1) * 3 * n_points] = {};
  float lmk_messages_lambda_ [(max_nlmkedges + 1) * 9 * n_points] = {};

  for (unsigned cID = 0; cID < n_keyframes; ++cID) {
    for (unsigned i = 0; i < cam_dofs; ++i) {
      cam_messages_eta_[(max_nkfedges + 1) * cam_dofs * cID + i] = cam_priors_eta_[cam_dofs * cID + i];
    }
    for (unsigned j = 0; j < cam_dofs * cam_dofs; ++j) {
      cam_messages_lambda_[(max_nkfedges + 1) * cam_dofs * cam_dofs * cID + j] = cam_priors_lambda_[cam_dofs * cam_dofs * cID + j];
    }
  }
  for (unsigned lID = 0; lID < n_points; ++lID) {
    for (unsigned i = 0; i < 3; ++i) {
      lmk_messages_eta_[(max_nlmkedges + 1) * 3 * lID + i] = lmk_priors_eta_[3 * lID + i];
    }
    for (unsigned j = 0; j < 9; ++j) {
      lmk_messages_lambda_[(max_nlmkedges + 1) * 9 * lID + j] = lmk_priors_lambda_[9 * lID + j];
    }
  }

  // Create buffers to hold beliefs when we stream them out
  float cam_beliefs_eta_ [n_keyframes * cam_dofs] = {};
  float cam_beliefs_lambda_ [n_keyframes * cam_dofs * cam_dofs] = {};
  float lmk_beliefs_eta_ [n_points * 3] = {};
  float lmk_beliefs_lambda_ [n_points * 3 * 3] = {};

  // float fout_eta_ [n_edges * (cam_dofs + 3)] = {};
  // float fout_lambda_ [n_edges * (cam_dofs + 3) * (cam_dofs + 3)] = {};

  // std::cout << "Number of keyframe nodes in the graph: " << n_keyframes << '\n';
  // std::cout << "Number of landmark nodes in the graph: " << n_points << '\n';
  // std::cout << "Number of edges in the graph: " << n_edges << '\n';
  // std::cout << "Number of tiles used: " << n_edges + n_keyframes + n_points << '\n';
  // std::cout << "Number of degrees of freedom at each keyframe node: " << cam_dofs << '\n';

  assert (measurements_camIDs.size() == n_edges);
  assert (measurements_lIDs.size() == n_edges);
  assert (measurements_.size() == 2*n_edges);

  // Compute the factor potentials in advance from the measurements and the measurement sigma
  // hpotentials_eta, hpotentials_lambda = compute_factor_potentials(measurements, sigmas)


  // Create the IPU device
  std::cout << "\nAttaching to IPU device" << std::endl;
  IPUModel ipuModel;
  auto dm = DeviceManager::createDeviceManager();
  auto hwDevices = dm.getDevices(TargetType::IPU, 8);
  // for (const auto &device : hwDevices) { device.attach();}
  auto &device = hwDevices.front();
  device.attach();
  std::cout << "\nAttached to IPU device" << std::endl;


  // Create the IPU model device
  // IPUModel ipuModel;
  // Device device = ipuModel.createDevice();

  // Create the Graph object 
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  graph.addCodelets("bp_codelets.cpp");

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
  DataStream instream_cam_messages_eta = graph.addHostToDeviceFIFO("instream_cam_messages_eta", FLOAT, (max_nkfedges + 1) * cam_dofs * n_keyframes);
  DataStream instream_cam_messages_lambda = graph.addHostToDeviceFIFO("instream_cam_messages_lambda", FLOAT, (max_nkfedges + 1) * cam_dofs * cam_dofs * n_keyframes);
  DataStream instream_lmk_messages_eta = graph.addHostToDeviceFIFO("instream_lmk_messages_eta", FLOAT, (max_nlmkedges + 1) * 3 * n_points);
  DataStream instream_lmk_messages_lambda = graph.addHostToDeviceFIFO("instream_lmk_messages_lambda", FLOAT, (max_nlmkedges + 1) * 9 * n_points);

  DataStream outstream_cam_messages_eta = graph.addDeviceToHostFIFO("outstream_cam_messages_eta", FLOAT, (max_nkfedges + 1) * cam_dofs * n_keyframes);


  // Create data streams to read out beliefs    
  DataStream outstream_cam_beliefs_eta = graph.addDeviceToHostFIFO("outstream_cam_beliefs_eta", FLOAT, n_keyframes * cam_dofs);
  DataStream outstream_cam_beliefs_lambda = graph.addDeviceToHostFIFO("outstream_cam_beliefs_lambda", FLOAT, n_keyframes * cam_dofs * cam_dofs);
  DataStream outstream_lmk_beliefs_eta = graph.addDeviceToHostFIFO("outstream_lmk_beliefs_eta", FLOAT, n_points * 3);
  DataStream outstream_lmk_beliefs_lambda = graph.addDeviceToHostFIFO("outstream_lmk_beliefs_lambda", FLOAT, n_points * 3 * 3);

  // Set tile mapping for keyframe nodes.
  for (unsigned cID = 0; cID < n_keyframes; ++cID) {
    // Set tile mapping for number of edges for a variable node.
    graph.setTileMapping(cams_dofs[cID], cID);
    // Set tile mapping for beliefs
    graph.setTileMapping(cam_beliefs_eta.slice(cID * cam_dofs, (cID + 1) * cam_dofs), cID);
    graph.setTileMapping(cam_beliefs_lambda.slice(cID * cam_dofs * cam_dofs, (cID + 1) * cam_dofs * cam_dofs), cID);

    graph.setTileMapping(cam_scaling[cID], cID);
    graph.setTileMapping(cam_weaken_flag[cID], cID);

    // Set tile mapping for messages
    graph.setTileMapping(cam_messages_eta.slice((max_nkfedges + 1) * cam_dofs * cID, (max_nkfedges + 1) * cam_dofs * (cID + 1)), cID);
    graph.setTileMapping(cam_messages_lambda.slice((max_nkfedges + 1) * cam_dofs * cam_dofs * cID, (max_nkfedges + 1) * cam_dofs * cam_dofs * (cID + 1) ), cID);
    graph.setTileMapping(pcam_messages_eta.slice((max_nkfedges + 1) * cam_dofs * cID, (max_nkfedges + 1) * cam_dofs * (cID + 1)), cID);
    graph.setTileMapping(pcam_messages_lambda.slice((max_nkfedges + 1) * cam_dofs * cam_dofs * cID, (max_nkfedges + 1) * cam_dofs * cam_dofs * (cID + 1) ), cID);
  }
  // Set tile mapping for landmark nodes.
  for (unsigned lID = 0; lID < n_points; ++lID) {
    // Set tile mapping for number of edges for a variable node.
    graph.setTileMapping(lmk_dofs[lID], lID + n_keyframes);
    // Set tile mapping for beliefs
    graph.setTileMapping(lmk_beliefs_eta.slice(lID * 3, (lID + 1) * 3), lID + n_keyframes);
    graph.setTileMapping(lmk_beliefs_lambda.slice(lID * 3 * 3, (lID + 1) * 3 * 3), lID + n_keyframes);

    graph.setTileMapping(lmk_scaling[lID], lID + n_keyframes);
    graph.setTileMapping(lmk_weaken_flag[lID], lID + n_keyframes);

    // Set tile mapping for messages
    graph.setTileMapping(lmk_messages_eta.slice((max_nlmkedges + 1) * 3 * lID, (max_nlmkedges + 1) * 3 * (lID + 1)), lID + n_keyframes);
    graph.setTileMapping(lmk_messages_lambda.slice((max_nlmkedges + 1) * 9 * lID, (max_nlmkedges + 1) * 9 * (lID + 1)), lID + n_keyframes);
    graph.setTileMapping(plmk_messages_eta.slice((max_nlmkedges + 1) * 3 * lID, (max_nlmkedges + 1) * 3 * (lID + 1)), lID + n_keyframes);
    graph.setTileMapping(plmk_messages_lambda.slice((max_nlmkedges + 1) * 9 * lID, (max_nlmkedges + 1) * 9 * (lID + 1)), lID + n_keyframes);
  }


  // CREATE FACTOR NODES FOR EACH EDGE

  // Tensors for regulating damping
  Tensor damping = graph.addVariable(FLOAT, {n_edges}, "damping");
  Tensor damping_count = graph.addVariable(INT, {n_edges}, "damping_count");
  Tensor mu = graph.addVariable(FLOAT, {n_edges * (cam_dofs + 3)}, "mu");
  Tensor oldmu = graph.addVariable(FLOAT, {n_edges * (cam_dofs + 3)}, "oldmu");
  Tensor dmu = graph.addVariable(FLOAT, {n_edges}, "dmu");

  Tensor active_flag = graph.addVariable(UNSIGNED_INT, {n_edges}, "active_flag");

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

  DataStream instream_measurements = graph.addHostToDeviceFIFO("instream_measurements", FLOAT, n_edges * 2);
  DataStream instream_meas_variances = graph.addHostToDeviceFIFO("instream_meas_variances", FLOAT, n_edges);
  DataStream instream_K = graph.addHostToDeviceFIFO("instream_K", FLOAT, n_edges * 9);

  DataStream outstream_factors_eta = graph.addDeviceToHostFIFO("outstream_factors_eta", FLOAT, n_edges * (cam_dofs + 3));
  DataStream outstream_factors_lambda = graph.addDeviceToHostFIFO("outstream_factors_lambda", FLOAT, n_edges * (cam_dofs + 3) * (cam_dofs + 3));

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

    unsigned tile = n_keyframes + n_points + edgeID;

    // First n_keyframes + n_points tiles are used for variable nodes, so use the next n_edges tiles for factor nodes.
    graph.setTileMapping(damping[edgeID], tile);
    graph.setTileMapping(damping_count[edgeID], tile);
    graph.setTileMapping(mu.slice(edgeID * (cam_dofs + 3), (edgeID + 1) * (cam_dofs + 3)), tile);
    graph.setTileMapping(oldmu.slice(edgeID * (cam_dofs + 3), (edgeID + 1) * (cam_dofs + 3)), tile);
    graph.setTileMapping(dmu[edgeID], tile);

    graph.setTileMapping(active_flag[edgeID], tile);

    graph.setTileMapping(measurements.slice(edgeID * 2, (edgeID + 1) * 2), tile);
    graph.setTileMapping(meas_variances[edgeID], tile);
    graph.setTileMapping(K.slice(edgeID * 9, (edgeID + 1) * 9), tile);

    graph.setTileMapping(factor_potentials_eta.slice(edgeID * (cam_dofs + 3), (edgeID + 1) * (cam_dofs + 3)), tile);
    graph.setTileMapping(factor_potentials_lambda.slice(edgeID * (cam_dofs + 3) * (cam_dofs + 3), (edgeID + 1) * (cam_dofs + 3) * (cam_dofs + 3)), tile);
    }

  auto csvec_computemessages = buildComputeMessagesCS(graph, n_keyframes, n_points, n_edges, cam_dofs, max_nkfedges, max_nlmkedges,
                                                  n_edges_per_kf, n_edges_per_lmk, measurements_camIDs,
                                                  measurements_lIDs, cams_dofs, lmk_dofs, active_flag, factor_potentials_eta, factor_potentials_lambda, 
                                                  cam_messages_eta, cam_messages_lambda, lmk_messages_eta, lmk_messages_lambda,
                                                  pcam_messages_eta, pcam_messages_lambda, plmk_messages_eta, plmk_messages_lambda,
                                                  cam_beliefs_eta, cam_beliefs_lambda, lmk_beliefs_eta, lmk_beliefs_lambda, 
                                                  measurements, meas_variances, K, damping, damping_count, mu, oldmu, dmu);

  auto cs_relinearise = buildRelineariseCS(graph, n_keyframes, n_points, n_edges, cam_dofs, measurements_camIDs, measurements_lIDs, 
                                          measurements, meas_variances, K, cam_beliefs_eta, cam_beliefs_lambda, lmk_beliefs_eta, lmk_beliefs_lambda,
                                          factor_potentials_eta, factor_potentials_lambda);

  auto cs_weaken_prior = buildWeakenPriorCS(graph, n_keyframes, n_points, cam_dofs, max_nkfedges, max_nlmkedges,
                                            cam_messages_eta, cam_messages_lambda, lmk_messages_eta, lmk_messages_lambda,
                                            cam_scaling, lmk_scaling, cam_weaken_flag, lmk_weaken_flag);

  auto cs_ub_slow = buildUpdateBeliefsCS(graph, n_keyframes, n_points, cam_dofs, cams_dofs, lmk_dofs, max_nkfedges, max_nlmkedges, n_edges_per_kf, n_edges_per_lmk, 
                                              cam_messages_eta, cam_messages_lambda, lmk_messages_eta, lmk_messages_lambda, 
                                              cam_beliefs_eta, cam_beliefs_lambda, lmk_beliefs_eta, lmk_beliefs_lambda);

  auto cs_ub_red = buildUpdateBeliefsProg(graph, 
                                n_keyframes,
                                n_points,
                                cam_dofs,
                                max_nkfedges,
                                max_nlmkedges,
                                cam_messages_eta,
                                cam_messages_lambda,
                                lmk_messages_eta,
                                lmk_messages_lambda,
                                cam_beliefs_eta,
                                cam_beliefs_lambda,
                                lmk_beliefs_eta,
                                lmk_beliefs_lambda);

  Sequence prog_ub;
  // prog_ub.add(PrintTensor("cam beliefs eta before updating", cam_beliefs_eta));
  // prog_ub.add(PrintTensor("cam beliefs eta after updating", cam_beliefs_eta.reshape({n_keyframes, cam_dofs})));
  // prog_ub.add(PrintTensor("cam messages used to update belief", cam_messages_eta.reshape({n_keyframes, max_nkfedges + 1, cam_dofs})[0]));
  // prog_ub.add(PrintTensor("cam messages used to update belief", cam_messages_lambda.reshape({n_keyframes, max_nkfedges + 1, 36})));
  for (const auto &cs : cs_ub_red) { prog_ub.add(Execute(cs));}  // Using reduction
  // prog_ub.add(Execute(cs_ub_slow));  // Slower method not using reductions
  // prog_ub.add(PrintTensor("cam beliefs eta after updating", cam_beliefs_eta.reshape({n_keyframes, cam_dofs})));
  // prog_ub.add(PrintTensor("cam beliefs lambda after updating", cam_beliefs_lambda.reshape({n_keyframes, 36})));

  Sequence prog_weaken_prior;
  // prog_weaken_prior.add(PrintTensor("cam0 lambda before weakening ", cam_messages_lambda[0]));
  // prog_weaken_prior.add(PrintTensor("lmk0 lambda before weakening ", lmk_messages_lambda[0]));
  prog_weaken_prior.add(Execute(cs_weaken_prior));
  // for (const auto &cs : cs_ub_red) { prog_weaken_prior.add(Execute(cs));}
  prog_weaken_prior.add(prog_ub);
  // prog_weaken_prior.add(PrintTensor("cam0 lambda after weakening ", cam_messages_lambda[0]));
  // prog_weaken_prior.add(PrintTensor("lmk0 lambda after weakening ", lmk_messages_lambda[0]));


  // Program to write initial data to tile
  Sequence wprog;
  wprog.add(Copy(instream_damping, damping));
  wprog.add(Copy(instream_damping_count, damping_count));
  wprog.add(Copy(instream_mu, mu));
  wprog.add(Copy(instream_oldmu, oldmu));
  wprog.add(Copy(instream_active_flag, active_flag));

  wprog.add(Copy(instream_cams_dofs, cams_dofs));
  wprog.add(Copy(instream_lmk_dofs, lmk_dofs));
  wprog.add(Copy(instream_cam_scaling, cam_scaling));
  wprog.add(Copy(instream_lmk_scaling, lmk_scaling));
  wprog.add(Copy(instream_cam_weaken_flag, cam_weaken_flag));
  wprog.add(Copy(instream_lmk_weaken_flag, lmk_weaken_flag));

  // Stream priors to tiles by streaming whole message tensor with zeros where measurement messages will be
  wprog.add(Copy(instream_cam_messages_eta, cam_messages_eta));
  wprog.add(Copy(instream_cam_messages_lambda, cam_messages_lambda));
  wprog.add(Copy(instream_lmk_messages_eta, lmk_messages_eta));
  wprog.add(Copy(instream_lmk_messages_lambda, lmk_messages_lambda));
  // Stream measurements to tiles
  wprog.add(Copy(instream_measurements, measurements));
  wprog.add(Copy(instream_meas_variances, meas_variances));
  wprog.add(Copy(instream_K, K));


  FloatingPointBehaviour behaviour(true, true, true, false, true);  // inv, div0, oflo, esr, nanoo

  Sequence prep_prog;
  // To throw errors when we have problems with floating point calculations
  setFloatingPointBehaviour(graph, prep_prog, behaviour, "");
  prep_prog.add(prog_ub);  // To send priors
  // int adj_cID = measurements_camIDs[0];
  // int adj_lID = measurements_lIDs[0];
  // prep_prog.add(PrintTensor("measurement", measurements.slice(0, 2)));
  // prep_prog.add(PrintTensor("meas_variance", meas_variances[0]));
  // prep_prog.add(PrintTensor("K", K.slice(0, 9)));
  // prep_prog.add(PrintTensor("kf_belief_eta_", cam_beliefs_eta.slice(adj_cID * cam_dofs, (adj_cID + 1) * cam_dofs)));
  // prep_prog.add(PrintTensor("kf_belief_lambda_", cam_beliefs_lambda.slice(adj_cID * cam_dofs * cam_dofs, (adj_cID + 1) * cam_dofs * cam_dofs)));
  // prep_prog.add(PrintTensor("lmk_belief_eta_", lmk_beliefs_eta.slice(adj_lID * 3, (adj_lID + 1) * 3)));
  // prep_prog.add(PrintTensor("lmk_belief_lambda_", lmk_beliefs_lambda.slice(adj_lID * 3 * 3, (adj_lID + 1) * 3 * 3)));

  prep_prog.add(Execute(cs_relinearise)); // To compute factors for the first time
  // prep_prog.add(PrintTensor("cam_beliefs_eta after priors are sent", cam_beliefs_eta.slice(0,12)));
  // prep_prog.add(PrintTensor("cam_beliefs_lambda after priors are sent", cam_beliefs_lambda.slice(0,72)));
  // prep_prog.add(PrintTensor("lmk_beliefs_eta after priors are sent", lmk_beliefs_eta.slice(0,9)));
  // prep_prog.add(PrintTensor("lmk_beliefs_lambda after priors are sent", lmk_beliefs_lambda.slice(0,27)));
  // prep_prog.add(PrintTensor("\nfactor_potentials_eta", factor_potentials_eta.slice(0,18)));
  // prep_prog.add(PrintTensor("\nfactor_potentials_lambda", factor_potentials_lambda.slice(0,162)));

  // prep_prog.add(PrintTensor("cam messages eta", cam_messages_eta));

  Sequence lprog;
  setFloatingPointBehaviour(graph, prep_prog, behaviour, "");
  lprog.add(Execute(csvec_computemessages[0]));  // Choose damping factor and relinearise if necessary
  // lprog.add(PrintTensor("\nmeans\n", mu.slice(0,18)));
  // lprog.add(PrintTensor("dmu values\n", dmu.slice(0,100)));
  // lprog.add(PrintTensor("damping values\n", damping.slice(0,100)));
  lprog.add(Copy(mu, oldmu));
  lprog.add(Execute(csvec_computemessages[1]));
  lprog.add(prog_ub);

  lprog.add(Copy(cam_messages_eta, pcam_messages_eta)); // Messages from factor node have been used so can be copied to previous message holder
  lprog.add(Copy(cam_messages_lambda, pcam_messages_lambda));
  lprog.add(Copy(lmk_messages_eta, plmk_messages_eta)); 
  lprog.add(Copy(lmk_messages_lambda, plmk_messages_lambda));



  // Program to stream out data from the IPU to the host
  Sequence rprog;
  rprog.add(Copy(cam_beliefs_eta, outstream_cam_beliefs_eta));
  rprog.add(Copy(cam_beliefs_lambda, outstream_cam_beliefs_lambda));
  rprog.add(Copy(lmk_beliefs_eta, outstream_lmk_beliefs_eta));
  rprog.add(Copy(lmk_beliefs_lambda, outstream_lmk_beliefs_lambda));

  // rprog.add(Copy(cam_messages_eta, outstream_cam_messages_eta));

  rprog.add(Copy(damping, outstream_damping));
  rprog.add(Copy(damping_count, outstream_damping_count));

  // rprog.add(Copy(factor_potentials_eta, outstream_factors_eta));
  // rprog.add(Copy(factor_potentials_lambda, outstream_factors_lambda));

  Sequence newkf;
  newkf.add(Copy(instream_active_flag, active_flag));
  newkf.add(Copy(instream_cam_weaken_flag, cam_weaken_flag));
  newkf.add(Copy(instream_lmk_weaken_flag, lmk_weaken_flag));


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
  timing_prog.add(wprog);
  timing_prog.add(prep_prog);
  timing_prog.add(lprog);
  timing_prog.add(change_prior);
  timing_prog.add(bulk_iters);


  // Create the engine
  OptionFlags engineOpts {
    {"debug.instrumentCompute", "true"},
    {"debug.computeInstrumentationLevel", "tile"},
    {"target.workerStackSizeInBytes", "4096"}
  };
  Engine engine(graph, {wprog, prep_prog, lprog, prog_weaken_prior, rprog, newkf}, engineOpts);
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

  engine.connectStream(instream_cam_messages_eta, &cam_messages_eta_[0]);
  engine.connectStream(instream_cam_messages_lambda, &cam_messages_lambda_[0]);
  engine.connectStream(instream_lmk_messages_eta, &lmk_messages_eta_[0]);
  engine.connectStream(instream_lmk_messages_lambda, &lmk_messages_lambda_[0]);
  engine.connectStream(instream_measurements, &measurements_[0]);
  engine.connectStream(instream_meas_variances, &meas_variances_[0]);
  engine.connectStream(instream_K, &K_[0]);

  engine.connectStream(outstream_cam_beliefs_eta, cam_beliefs_eta_);
  engine.connectStream(outstream_cam_beliefs_lambda, cam_beliefs_lambda_);
  engine.connectStream(outstream_lmk_beliefs_eta, lmk_beliefs_eta_);
  engine.connectStream(outstream_lmk_beliefs_lambda, lmk_beliefs_lambda_);

  // engine.connectStream(outstream_factors_eta, fout_eta_);
  // engine.connectStream(outstream_factors_lambda, fout_lambda_);

  engine.connectStream(outstream_cam_messages_eta, cam_messages_eta_);
  engine.connectStream(outstream_damping, &damping_[0]);
  engine.connectStream(outstream_damping_count, &damping_count_[0]);

  // Run programs
  Engine::TimerTimePoint time0 = engine.getTimeStamp();
  auto start_time = std::chrono::high_resolution_clock::now();   // Time how long the message passing program takes

  std::cout << "Running program to stream initial data to IPU\n";
  engine.run(0);
  std::cout << "Initial data streaming complete\n\n";

  std::cout << "Sending priors and computing factor potentials.\n";
  engine.run(1);

  engine.run(4);
  string fce = "res/" + dir + "/beliefs/cb_eta0.txt";
  string fcl = "res/" + dir + "/beliefs/cb_lambda0.txt";
  string fle = "res/" + dir + "/beliefs/lb_eta0.txt";
  string fll = "res/" + dir + "/beliefs/lb_lambda0.txt";
  saveFloatVector(fce, &cam_beliefs_eta_[0], cam_dofs * n_keyframes);
  saveFloatVector(fcl, &cam_beliefs_lambda_[0], cam_dofs * cam_dofs * n_keyframes);
  saveFloatVector(fle, &lmk_beliefs_eta_[0], 3 * n_points);
  saveFloatVector(fll, &lmk_beliefs_lambda_[0], 9 * n_points);


  ofstream reproj_file;
  ofstream cost_file;
  ofstream num_relins_file;
  reproj_file.open("res/" + dir + "/ipu_reproj.txt");
  cost_file.open("res/" + dir + "/ipu_cost.txt");
  num_relins_file.open("res/" + dir + "/num_relins.txt");

  unsigned data_counter = 0;
  float reproj [2] = {};
  eval_reproj(reproj, n_edges, active_flag_, data_counter, cam_beliefs_eta_, cam_beliefs_lambda_, lmk_beliefs_eta_, lmk_beliefs_lambda_,
                      &measurements_camIDs[0], &measurements_lIDs[0], &measurements_[0], &Ksingle_[0]);
  cout << "Initial Reprojection error " << reproj[0] << " Cost " << reproj[1] << "\n";
  reproj_file << reproj[0] << endl;
  cost_file << reproj[1] << endl;
  num_relins_file << 0 << endl;

  int num_relins = 0;
  unsigned n_iters_per_newkf = 200;
  unsigned niters = (n_keyframes - 1) * n_iters_per_newkf;
  unsigned iter = 0;
  for (unsigned i = 0; i < (niters); ++i) {
    std::cout << "\nIteration: " << i << '\n';

    if ((i+1) % n_iters_per_newkf == 0) {
      cout << "********************** Adding new keyframe **************************\n";
      engine.run(5);
      iter = 0;
      data_counter += 1;


      // // Use previous kf for prior on next keyframe
      // Matrix<float,6,1> previous_kf_eta = Map<Matrix<float,6,1>>(&cam_beliefs_eta_[data_counter * 6]);
      // Matrix<float,6,6> previous_kf_lam = Map<Matrix<float,6,6>>(&cam_beliefs_lambda_[data_counter * 36]);
      // VectorXf previous_kf_mu = previous_kf_lam.transpose().inverse() * previous_kf_eta;
      // Matrix<float,6,6> new_kf_lam = Map<Matrix<float,6,6>>(&cam_priors_lambda_[(data_counter + 1) * 36]);
      // VectorXf new_kf_eta = new_kf_lam.transpose() * previous_kf_mu;
      // for (unsigned i = 0; i < 6; ++i) {
      //   cam_priors_eta_[(data_counter + 1) * 6 + i] = new_kf_eta(i);
      // }

      // // Use prior on keyframe for prior on newly observed landmarks
      // Vector3f previous_kf_w;
      // previous_kf_w << previous_kf_mu(3), previous_kf_mu(4), previous_kf_mu(5);
      // Matrix3f previous_kf_R_w2c = eigenso3exp(previous_kf_w);
      // Vector4f loc_cam_frame;
      // loc_cam_frame << 0.0, 0.0, 1.0, 1.0;
      // Matrix4f Tw2c;
      // Tw2c << Rw2c(0,0), Rw2c(0,1), Rw2c(0,2), previous_kf_mu(0),
      //         Rw2c(1,0), Rw2c(1,1), Rw2c(1,2), previous_kf_mu(1),
      //         Rw2c(2,0), Rw2c(2,1), Rw2c(2,2), previous_kf_mu(2),
      //         0.0, 0.0, 0.0, 1.0;
      // Vector3f new_lmk_mu_wf = Tw2c.inverse() * loc_cam_frame;

      // for (unsigned i = 0; i < n_points; ++i) {
      //   if (lmk_weaken_flag[data_counter*n_points + i] == 5) {  // then newly observed landmark


      //   }
      // }

    }

    if (((iter+ 1) % 2 == 0) && (iter < num_prior_weak_steps * 2)) {
      cout << "Weakening priors \n";
      engine.run(3);
    }

    engine.run(2); // synchronous update
    engine.run(4); // Write messages out

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
    eval_reproj(reproj, n_edges, active_flag_, data_counter, 
                cam_beliefs_eta_, cam_beliefs_lambda_, lmk_beliefs_eta_, lmk_beliefs_lambda_,
                &measurements_camIDs[0], &measurements_lIDs[0], &measurements_[0], &Ksingle_[0]);
    cout << "Reprojection error " << reproj[0] << " // Cost " << reproj[1] << " // Number of relinearisations: " << num_relins << "\n";
    reproj_file << reproj[0] << endl;
    cost_file << reproj[1] << endl;
    num_relins_file << num_relins << endl;

    string fce = "res/" + dir + "/beliefs/cb_eta" + std::to_string(i+1) + ".txt";
    string fcl = "res/" + dir + "/beliefs/cb_lambda" + std::to_string(i+1) + ".txt";
    string fle = "res/" + dir + "/beliefs/lb_eta" + std::to_string(i+1) + ".txt";
    string fll = "res/" + dir + "/beliefs/lb_lambda" + std::to_string(i+1) + ".txt";
    saveFloatVector(fce, &cam_beliefs_eta_[0], cam_dofs * n_keyframes);
    saveFloatVector(fcl, &cam_beliefs_lambda_[0], cam_dofs * cam_dofs * n_keyframes);
    saveFloatVector(fle, &lmk_beliefs_eta_[0], 3 * n_points);
    saveFloatVector(fll, &lmk_beliefs_lambda_[0], 9 * n_points);


    // Print beliefs that are streamed out 
    // std::cout << "\nKeyframe Eta beliefs: \n";
    // for (unsigned i = 0; i <  24; ++i) {
    //   printf("%.12f  ", cam_beliefs_eta_[i]);  
    // };
    // std::cout << "\nKeyframe Lambda beliefs: \n";
    // for (unsigned i = 0; i <  36; ++i) {
    //   printf("%.12f  ", cam_beliefs_lambda_[i]);  
    // }
    // std::cout << '\n';

    // std::cout << "\nLandmark Eta beliefs: \n";
    // for (unsigned i = 0; i <  12; ++i) {
    //   printf("%.12f  ", lmk_beliefs_eta_[i]);  
    // }
    // std::cout << "\nLandmark Lambda beliefs: \n";
    // for (unsigned i = 0; i <  3 * 3 * 2; ++i) {
    //   printf("%.12f  ", lmk_beliefs_lambda_[i]);  
    // }
    // std::cout << '\n';
    iter += 1;
  }

  // reproj_file.close();
  // cost_file.close();
  // num_relins_file.close();

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> elapsed = end_time - start_time;
  // std::cout << "\nElapsed Time  on chrono for " << niters << " synchronous updates: " << elapsed.count() << " micro seconds" << "\n";
  
  // Engine::TimerTimePoint time1 = engine.getTimeStamp();
  // string timing_report = engine.reportTiming(time0, time1);
  // std::cout << timing_report << "\n";

  // // Timing program
  // Engine::TimerTimePoint time0 = engine.getTimeStamp();
  // auto start_time = std::chrono::high_resolution_clock::now();   // Time how long the message passing program takes
  // std::cout << "Executing timing program\n";
  // engine.run(0);
  // auto end_time = std::chrono::high_resolution_clock::now();
  // std::chrono::duration<double, std::micro> elapsed = end_time - start_time;
  // std::cout << "\nElapsed Time for timing program: " << elapsed.count() << " micro seconds" << "\n";
  // Engine::TimerTimePoint time1 = engine.getTimeStamp();
  // string timing_report = engine.reportTiming(time0, time1);
  // std::cout << timing_report << "\n";
  // // engine.run(1);



  // // Profiling
  // char* log_dir = std::getenv("GC_PROFILE_LOG_DIR");
  // std::string direc = log_dir ? std::string(log_dir) : ".";

  // // Save intervals report
  // std::ofstream fout(direc + "/intervals.csv");
  // engine.reportIntervals(fout);

  // // Graph Report
  // poplar::ProfileValue graphProfile = engine.getGraphProfile();
  // std::ofstream graphReport;
  // graphReport.open(direc + "/graph.json");
  // poplar::serializeToJSON(graphReport, graphProfile);
  // graphReport.close();

  // // Execution Report
  // poplar::ProfileValue execProfile = engine.getExecutionProfile();
  // std::ofstream execReport;
  // execReport.open(direc + "/execution.json");
  // poplar::serializeToJSON(execReport, execProfile);
  // execReport.close();

  // const auto &cycles = execProfile["computeSetCyclesByTile"].asVector();
  // int cscId = cs_ub_red.getId();
  // auto numtiles = device.getTarget().getNumTiles();
  // for (auto t=0;t<numtiles;++t) {
  //   auto camtileCycles = cycles[cscId][t].asInt();
  //   cout << t << " tile cycle " << camtileCycles << "\n";
  // }

  return 0;
}
