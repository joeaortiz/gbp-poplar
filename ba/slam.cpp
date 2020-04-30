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

#include "../include/dataio.h"
#include "../include/util.h"

using namespace std;
using namespace poplar;
using namespace poplar::program;
using namespace popops;
using namespace Eigen;


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
  std::string bal_file;
  int nIPUs;
  bool profile;
  int iters_between_kfs;
  int cams_per_tile;
  float transnoise;
  float rotnoise;
  float lmktrans_noise;
  bool av_depth_on;
  float av_depth;
  float reproj_meas_var;
  float prior_std_weaker_factor;
  float first_cam_prior_std;
  float steps;
  int iters_before_damping;
  bool verbose;
};

Options parseOptions(int argc, char** argv) {
  Options options;
  std::string modeString;

  namespace po = boost::program_options;
  po::options_description desc("Options");
  desc.add_options()
  ("help", "Show command help")
  ("bal_file",
   po::value<std::string>(&options.bal_file)->required(),
   "Set the bal file"
  )
  ("ipus",
   po::value<int>(&options.nIPUs)->default_value(1),
   "Number of IPU chips to use. NB. Must be a power of 2!"
  )
  ("profile",
   po::value<bool>(&options.profile)->default_value(false),
   "Save profile report after execution"
  )
  ("iters_between_kfs",
   po::value<int>(&options.iters_between_kfs)->default_value(700),
   "Number of GBP iterations between sucessive keyframes."
  )
  ("camspertile",
   po::value<int>(&options.cams_per_tile)->default_value(1),
   "Maximum number of keyframe nodes placed on a single tile of the IPU."
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
  ("avdepth_on",
   po::value<bool>(&options.av_depth_on)->default_value(false),
   "bool: should landmarks be initialised at an average depth from the keyframe they are first observed by"
  )
  ("avdepth",
   po::value<float>(&options.av_depth)->default_value(1.f),
   "float: Average depth at which landmarks are initialed from the keyframe which they are first observed by."
  )
  ("reproj_meas_var",
   po::value<float>(&options.reproj_meas_var)->default_value(4.f),
   "Variance of Gaussian noise in Gaussian measurement model for the reprojection constraints"
  )
  ("prior_std_weaker_factor",
   po::value<float>(&options.prior_std_weaker_factor)->default_value(100.f),
   "Factor: std of gauss noise of reprojection factors / std of gauss noise of prior factors"
  )
  ("first_cam_prior_std",
   po::value<float>(&options.first_cam_prior_std)->default_value(0.01f),
   "Standard deviation of prior on pose of first keyframe, to anchor optimisation."
  )
  ("steps",
   po::value<float>(&options.steps)->default_value(5.f),
   "The priors are gradually weakened over this many steps."
  )
  ("undamped_start",
   po::value<int>(&options.iters_before_damping)->default_value(15),
   "Number of undamped iterations before damping GBP."
  )
  ("v",
   po::value<bool>(&options.verbose)->default_value(false),
   "Verbose: print beliefs"
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
  const char * f = options.bal_file.c_str();

  BALProblem bal_problem;
  if (!bal_problem.LoadFile(f)) {  //true for load planes and timestamps
    std::cerr << "ERROR: unable to open file " << f << "\n";
    return 1;
  }

  // Extract data from BALProblem object
  unsigned n_edges = bal_problem.n_edges();  
  unsigned n_keyframes = bal_problem.n_keyframes();
  unsigned n_points = bal_problem.n_points();

  std::vector<float> Ksingle_ {(float)(bal_problem.fx()[0]), 0.f, (float)(bal_problem.cx()[0]), 0.f, 
    (float)(bal_problem.fy()[0]), (float)(bal_problem.cy()[0]), 0.f, 0.f, 1.f};
  float K_ [n_edges * 9] = {};
  for (unsigned i = 0; i < n_edges; ++i) {
    for (unsigned j = 0; j < 9; ++j) {
      K_[i*9 + j] = Ksingle_[j];
    }
  }

  std::vector<float> meas_variances_ (n_edges, options.reproj_meas_var);
  std::vector<float> measurements_;
  std::vector<unsigned> measurements_camIDs;
  std::vector<unsigned> measurements_lIDs;
  for (int i = 0; i < bal_problem.n_edges(); ++i) {
    measurements_.push_back(bal_problem.observations()[2*i]);
    measurements_.push_back(bal_problem.observations()[2*i+1]);
    measurements_camIDs.push_back(bal_problem.camera_index(i));
    measurements_lIDs.push_back(bal_problem.point_index(i));
  }

  std::vector<unsigned> n_edges_per_kf;
  for (int i = 0; i < bal_problem.n_keyframes(); ++i) {
    n_edges_per_kf.push_back(std::count(measurements_camIDs.begin(), measurements_camIDs.end(), i));
  }
  std::vector<unsigned> n_edges_per_lmk;
  for (int i = 0; i < bal_problem.n_points(); ++i) {
    n_edges_per_lmk.push_back(std::count(measurements_lIDs.begin(), measurements_lIDs.end(), i));
  }

  std::vector<float> cam_priors_mean_;
  for (int i = 0; i < bal_problem.n_keyframes(); ++i) {
    for (int j = 0; j<6; ++j) {
      cam_priors_mean_.push_back((float)(bal_problem.camera(i)[j]));
    }
  }
  std::vector<float> lmk_priors_mean_;
  for (int i = 0; i < bal_problem.n_points(); ++i) {
    for (int j = 0; j<3; ++j) {
      lmk_priors_mean_.push_back((float)(bal_problem.point(i)[j]));
    }
  }

  // Add noise to initialisation here if desired
  if (options.transnoise != 0.f) {
    add_cam_trans_noise(cam_priors_mean_, n_keyframes, options.transnoise);
  }
  if (options.rotnoise != 0.f) {
    add_cam_rot_noise(cam_priors_mean_, n_keyframes, options.rotnoise);
  }
  if ((options.lmktrans_noise != 0.f) && !(options.av_depth_on)) {
    add_lmk_noise(lmk_priors_mean_, n_points, options.lmktrans_noise);
  } else if (options.av_depth_on) {
    av_depth_init(options.av_depth, n_keyframes, n_edges, n_points,
                  measurements_camIDs, measurements_lIDs, cam_priors_mean_, lmk_priors_mean_);
  }

  // Set priors to initially be similar strength to measurement factors
  std::vector<float> cam_priors_lambda_(bal_problem.n_keyframes() * 36, 0);
  std::vector<float> cam_priors_eta_(bal_problem.n_keyframes() * 6, 0);
  std::vector<float> lmk_priors_lambda_(bal_problem.n_points() * 9, 0);
  std::vector<float> lmk_priors_eta_(bal_problem.n_points() * 3, 0);
  set_prior_lambda(bal_problem, Ksingle_, options.reproj_meas_var, 
                  cam_priors_mean_, cam_priors_eta_, cam_priors_lambda_,
                  lmk_priors_mean_, lmk_priors_eta_, lmk_priors_lambda_);


  // Compute factor to scale lambda and eta when weakening the prior over iterations
  float cam_scaling_ [n_keyframes] = {};
  for (unsigned cID = 0; cID < n_keyframes; ++cID) {
    if (cID == 0 || cID == 1) {
      cam_scaling_[cID] = exp(-1 / options.steps * log(cam_priors_lambda_[cID*36] * pow(options.first_cam_prior_std, 2)));
    } else {
      cam_scaling_[cID] = exp(-2 / options.steps * log(options.prior_std_weaker_factor));
    }
  }
  float lmk_scaling_ [n_points] = {};
  for (unsigned lID = 0; lID < n_points; ++lID) {
    lmk_scaling_[lID] = exp(-2 / options.steps * log(options.prior_std_weaker_factor));
  }

  std::cout << "Loaded data onto host!\n";

  unsigned cam_dofs = 6;
  std::vector<unsigned> cams_dofs_(n_keyframes, 6);
  std::vector<unsigned> lmk_dofs_(n_points, 3);

  std::vector<float> damping_(n_edges, 0.0);
  std::vector<int> damping_count_(n_edges, -options.iters_before_damping);
  std::vector<float> mu_(n_edges*(cam_dofs + 3), 0.0);
  std::vector<float> oldmu_(n_edges*(cam_dofs + 3), 0.0);
  std::vector<unsigned> robust_flag_(n_edges, 0);

  cout << "SLAM\n";

  unsigned steps = static_cast<unsigned>(options.steps);
  std::vector<unsigned int> active_flag_ (bal_problem.n_edges(), 0);  // edge active flag
  std::vector<unsigned int> cam_weaken_flag_ (bal_problem.n_keyframes(), 0);
  std::vector<unsigned int> lmk_weaken_flag_ (bal_problem.n_points(), 0);
  std::vector<unsigned int> lmk_active_flag (bal_problem.n_points(), 0); 
  create_flags(bal_problem, active_flag_, cam_weaken_flag_, lmk_weaken_flag_,
               lmk_active_flag, steps);

  std::vector<unsigned> bad_associations;// = readUnsignedIntVector(dir + "/bad_associations.txt"); 

  // Create data buffers for streaming to and from the IPU
  unsigned max_nkfedges = *max_element(n_edges_per_kf.begin(), n_edges_per_kf.end());
  unsigned max_nlmkedges = *max_element(n_edges_per_lmk.begin(), n_edges_per_lmk.end());

  float cam_beliefs_eta_ [n_keyframes * cam_dofs] = {};
  float cam_beliefs_lambda_ [n_keyframes * cam_dofs * cam_dofs] = {};
  float lmk_beliefs_eta_ [n_points * 3] = {};
  float lmk_beliefs_lambda_ [n_points * 3 * 3] = {};

  std::cout << "\nNumber of keyframe nodes in the graph: " << n_keyframes << '\n';
  std::cout << "Number of landmark nodes in the graph: " << n_points << '\n';
  std::cout << "Number of edges in the graph: " << n_edges << '\n';

  // Allocate factor graph to tiles of IPU
  int nIPUs = options.nIPUs;
  if (options.nIPUs == 0) { nIPUs = 1;}
  if (options.nIPUs == 3) { nIPUs = 4;}
  if ((4<options.nIPUs) && (options.nIPUs<8)) { nIPUs = 8;}
  if ((8<options.nIPUs) && (options.nIPUs<16)) { nIPUs = 16;}

  int n_tiles = nIPUs * 1216;
  unsigned cams_per_tile = options.cams_per_tile;
  unsigned n_cam_tiles = ceil((float)n_keyframes / (float)cams_per_tile);
  float remaining_tiles = (float)(n_tiles - n_cam_tiles);
  unsigned lmks_per_tile = ceil((float)n_points / remaining_tiles);
  unsigned factors_per_tile = ceil((float)n_edges / remaining_tiles);

  unsigned n_lmk_tiles = ceil((float)n_points / (float)lmks_per_tile);
  unsigned n_factor_tiles = ceil((float)n_edges / (float)factors_per_tile);
  
  std::cout << "\nNumber of IPUs: " << nIPUs << '\n';
  cout << "Number of keyframe / landmark / factor nodes per tile: ";
  cout << cams_per_tile << " / " << lmks_per_tile << " / " << factors_per_tile  << "\n";
  cout << "Number of tiles used by camera / landmark / factor nodes / total : " << n_cam_tiles << " / " << n_lmk_tiles << " / " << n_factor_tiles << " / " << n_cam_tiles + n_factor_tiles << "\n"; 

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
  std::cout << "\nAttached to device: " << device.getId() << std::endl;

  // if (device.getId()) {
  //  std::cout << "Could not find a device\n";
  //  exit(-1);
  // }


  // Create the Graph object 
  Graph graph(device.getTarget());
  popops::addCodelets(graph);
  graph.addCodelets("gbp_codelets.cpp");

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


  // Set tile mapping for keyframe nodes.
  unsigned cam_tile = 0;
  for (unsigned cID = 0; cID < n_keyframes; ++cID) {
    cam_tile = floor((float)cID / (float)cams_per_tile);
    graph.setTileMapping(cams_dofs[cID], cam_tile);
    graph.setTileMapping(cam_beliefs_eta.slice(cID * cam_dofs, (cID + 1) * cam_dofs), cam_tile);
    graph.setTileMapping(cam_beliefs_lambda.slice(cID * cam_dofs * cam_dofs, (cID + 1) * cam_dofs * cam_dofs), cam_tile);

    graph.setTileMapping(cam_scaling[cID], cam_tile);
    graph.setTileMapping(cam_weaken_flag[cID], cam_tile);

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
    graph.setTileMapping(lmk_dofs[lID], lmk_tile + n_cam_tiles);
    graph.setTileMapping(lmk_beliefs_eta.slice(lID * 3, (lID + 1) * 3), lmk_tile + n_cam_tiles);
    graph.setTileMapping(lmk_beliefs_lambda.slice(lID * 3 * 3, (lID + 1) * 3 * 3), lmk_tile + n_cam_tiles);

    graph.setTileMapping(lmk_scaling[lID], lmk_tile + n_cam_tiles);
    graph.setTileMapping(lmk_weaken_flag[lID], lmk_tile + n_cam_tiles);

    // Store zero padding part of messages on variable node tile
    graph.setTileMapping(lmk_messages_eta.slice((max_nlmkedges + 1) * 3 * lID + n_edges_per_lmk[lID] * 3, (max_nlmkedges + 1) * 3 * (lID + 1)), lmk_tile + n_cam_tiles);
    graph.setTileMapping(lmk_messages_lambda.slice((max_nlmkedges + 1) * 9 * lID + n_edges_per_lmk[lID] * 9, (max_nlmkedges + 1) * 9 * (lID + 1)), lmk_tile + n_cam_tiles);
    graph.setTileMapping(plmk_messages_eta.slice((max_nlmkedges + 1) * 3 * lID + n_edges_per_lmk[lID] * 3, (max_nlmkedges + 1) * 3 * (lID + 1)), lmk_tile + n_cam_tiles);
    graph.setTileMapping(plmk_messages_lambda.slice((max_nlmkedges + 1) * 9 * lID + n_edges_per_lmk[lID] * 9, (max_nlmkedges + 1) * 9 * (lID + 1)), lmk_tile + n_cam_tiles);
  }


  // CREATE FACTOR NODES FOR EACH EDGE

  // Tensors for controlling damping
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

  unsigned count_edges_per_kf [n_keyframes] = {};
  unsigned count_edges_per_lmk [n_points] = {};

  unsigned edge_tile = 0;
  for (unsigned edgeID = 0; edgeID < n_edges; ++edgeID) {
    unsigned adj_cID = measurements_camIDs[edgeID];
    unsigned adj_lID = measurements_lIDs[edgeID];

    edge_tile = n_cam_tiles + floor((float)edgeID / (float)factors_per_tile);

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
  for (const auto &cs : cs_ub_red) { prog_ub.add(Execute(cs));}  // Using reduction

  Sequence prog_weaken_prior;
  prog_weaken_prior.add(Execute(cs_weaken_prior));
  prog_weaken_prior.add(prog_ub);

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

  Sequence gbp_iter_prog;
  setFloatingPointBehaviour(graph, linearise_prog, behaviour, "");
  gbp_iter_prog.add(Execute(csvec_computemessages[0]));  // Choose damping factor and relinearise if necessary
  gbp_iter_prog.add(Copy(mu, oldmu));
  gbp_iter_prog.add(Execute(csvec_computemessages[1]));
  gbp_iter_prog.add(prog_ub);
  // Messages from factor node have been used so can be copied to previous message holder
  gbp_iter_prog.add(Copy(cam_messages_eta, pcam_messages_eta)); 
  gbp_iter_prog.add(Copy(cam_messages_lambda, pcam_messages_lambda));
  gbp_iter_prog.add(Copy(lmk_messages_eta, plmk_messages_eta)); 
  gbp_iter_prog.add(Copy(lmk_messages_lambda, plmk_messages_lambda));

  // Program to stream out data from the IPU to the host
  Sequence read_prog;
  read_prog.add(Copy(cam_beliefs_eta, outstream_cam_beliefs_eta));
  read_prog.add(Copy(cam_beliefs_lambda, outstream_cam_beliefs_lambda));
  read_prog.add(Copy(lmk_beliefs_eta, outstream_lmk_beliefs_eta));
  read_prog.add(Copy(lmk_beliefs_lambda, outstream_lmk_beliefs_lambda));
  read_prog.add(Copy(damping, outstream_damping));
  read_prog.add(Copy(damping_count, outstream_damping_count));
  read_prog.add(Copy(robust_flag, outstream_robust_flag));

  Sequence read_priors_prog;
  read_priors_prog.add(Copy(cam_messages_eta.reshape({n_keyframes, max_nkfedges + 1, cam_dofs}).slice({0, 0}, {n_keyframes, 1}), outstream_cam_priors_eta));
  read_priors_prog.add(Copy(cam_messages_lambda.reshape({n_keyframes, max_nkfedges + 1, cam_dofs*cam_dofs}).slice({0, 0}, {n_keyframes, 1}), outstream_cam_priors_lambda));
  read_priors_prog.add(Copy(lmk_messages_eta.reshape({n_points, max_nlmkedges + 1, 3}).slice({0, 0}, {n_points, 1}), outstream_lmk_priors_eta));
  read_priors_prog.add(Copy(lmk_messages_lambda.reshape({n_points, max_nlmkedges + 1, 9}).slice({0, 0}, {n_points, 1}), outstream_lmk_priors_lambda));

  Sequence new_kf_prog;
  new_kf_prog.add(Copy(instream_damping_count, damping_count));
  new_kf_prog.add(Copy(instream_cam_priors_eta, cam_messages_eta.reshape({n_keyframes, max_nkfedges + 1, cam_dofs}).slice({0, 0}, {n_keyframes, 1}) ));
  new_kf_prog.add(Copy(instream_cam_priors_lambda, cam_messages_lambda.reshape({n_keyframes, max_nkfedges + 1, cam_dofs*cam_dofs}).slice({0, 0}, {n_keyframes, 1}) ));
  new_kf_prog.add(Copy(instream_lmk_priors_eta, lmk_messages_eta.reshape({n_points, max_nlmkedges + 1, 3}).slice({0, 0}, {n_points, 1}) ));
  new_kf_prog.add(Copy(instream_lmk_priors_lambda, lmk_messages_lambda.reshape({n_points, max_nlmkedges + 1, 9}).slice({0, 0}, {n_points, 1}) ));
  new_kf_prog.add(Copy(instream_active_flag, active_flag));
  new_kf_prog.add(Copy(instream_cam_weaken_flag, cam_weaken_flag));
  new_kf_prog.add(Copy(instream_lmk_weaken_flag, lmk_weaken_flag));
  new_kf_prog.add(prog_ub);  // Update belief of new nodes

  // Create the engine
  OptionFlags engineOpts {
    {"debug.instrumentCompute", "true"},
    {"debug.computeInstrumentationLevel", "tile"},
    {"target.workerStackSizeInBytes", "4096"}
  };

  enum ProgamIds {
    WRITE_PROG = 0,
    LINEARISE_PROG,
    GBP_PROG,
    WEAKEN_PRIORS,
    READ_PROG,
    READ_PRIORS,
    NEW_KEYFRAME,
    NUM_PROGS
  };
  std::vector<poplar::program::Program> progs =
    {write_prog, linearise_prog, gbp_iter_prog, prog_weaken_prior, read_prog, read_priors_prog, new_kf_prog};

  Engine engine(graph, progs, engineOpts);
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
  engine.connectStream(instream_cam_weaken_flag, &cam_weaken_flag_[0]);
  engine.connectStream(instream_lmk_weaken_flag, &lmk_weaken_flag_[0]);
  engine.connectStream(instream_active_flag, &active_flag_[0]);

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

  engine.connectStream(outstream_cam_priors_eta, &cam_priors_eta_[0]);
  engine.connectStream(outstream_cam_priors_lambda, &cam_priors_lambda_[0]);
  engine.connectStream(outstream_lmk_priors_eta, &lmk_priors_eta_[0]);
  engine.connectStream(outstream_lmk_priors_lambda, &lmk_priors_lambda_[0]);

  engine.connectStream(outstream_robust_flag, &robust_flag_[0]);

  // Run programs
  Engine::TimerTimePoint time0 = engine.getTimeStamp();
  std::cout << "Running program to stream initial data to IPU\n";
  engine.run(WRITE_PROG);
  std::cout << "Initial data streaming complete\n\n";

  std::cout << "Sending priors and computing factor potentials.\n";
  engine.run(LINEARISE_PROG);
  engine.run(READ_PROG);

  float reproj [2] = {};
  unsigned data_counter = 0;
  unsigned nrobust_edges = 0;
  int num_relins = 0;
  int n_new_lmks = 0;
  eval_reprojection_error(reproj, n_edges, active_flag_, cam_beliefs_eta_, 
                          cam_beliefs_lambda_, lmk_beliefs_eta_, lmk_beliefs_lambda_,
                          &measurements_camIDs[0], &measurements_lIDs[0], 
                          &measurements_[0], &Ksingle_[0], bad_associations);
  std::cout << "Initial Reprojection error: " << reproj[0] << " Cost " << reproj[1] << "\n";

  unsigned niters = (n_keyframes - 1) * options.iters_between_kfs - 1;
  unsigned iter = 0;
  std::cout << "Total number of GBP iterations: " << niters << "\n";
  std::cout << "GBP iterations between sucessive keyframes: " << options.iters_between_kfs << "\n";

  for (unsigned i = 0; i < (niters); ++i) {

      if ((i+1) % options.iters_between_kfs == 0) {
        iter = 0;
        data_counter += 1;  // Number of keyframes = data counter + 2

        n_new_lmks = update_flags(bal_problem, active_flag_, lmk_weaken_flag_, 
                     cam_weaken_flag_, lmk_active_flag, steps, data_counter);

        std::cout << "\n**********************************************************";
        std::cout << "\n Adding keyframe " << data_counter + 1;
        std::cout << "\n Adding " << n_new_lmks << " new landmarks";
        std::cout << "\n**********************************************************\n\n";

        // Use previous kf for prior on next keyframe
        engine.run(READ_PRIORS); // Read current priors back to host

        initialise_new_kf(cam_priors_eta_, lmk_priors_eta_, cam_beliefs_eta_, 
                          cam_beliefs_lambda_, cam_priors_lambda_, lmk_priors_lambda_,
                          lmk_weaken_flag_, data_counter, n_points);

        for (unsigned i = 0; i < n_edges; ++i) {
          damping_count_[i] = -15;
        }

        engine.run(NEW_KEYFRAME);
        engine.run(READ_PROG);

      }


    if (((iter+ 1) % 2 == 0) && (iter < options.steps * 2)) {
      cout << "Weakening priors \n";
      engine.run(WEAKEN_PRIORS);
    }

    engine.run(GBP_PROG); // synchronous update
    engine.run(READ_PROG); // Write beliefs out

    nrobust_edges = 0;
    for (unsigned i = 0; i < n_edges; ++i) {
      nrobust_edges += robust_flag_[i];
    }

    num_relins = 0;
    for (unsigned i = 0; i < n_edges; ++i) {
      if (damping_count_[i] == -8) {
        num_relins += 1;
      }
    }

    eval_reprojection_error(reproj, n_edges, active_flag_, cam_beliefs_eta_,
                            cam_beliefs_lambda_, lmk_beliefs_eta_, lmk_beliefs_lambda_,
                            &measurements_camIDs[0], &measurements_lIDs[0], 
                            &measurements_[0], &Ksingle_[0], bad_associations);
    cout << "Iters " << options.iters_between_kfs*data_counter + iter;
    cout << " (since last kf " << iter << ") // Reprojection error " << reproj[0];
    cout << " // Cost " << reproj[1] << " // n relins: " << num_relins;
    cout << " // n robust edges " << nrobust_edges << "\n";


    if (options.verbose) {
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

    iter += 1;
  }
  std::cout << "\n Finished GBP.\n";

  Engine::TimerTimePoint time1 = engine.getTimeStamp();
  string timing_report = engine.reportTiming(time0, time1);
  std::cout << timing_report << "\n";

  if (options.profile) {
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
  }

  return 0;
}
