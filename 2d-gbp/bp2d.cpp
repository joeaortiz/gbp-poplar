// Implementation of BP with single type of variable node with 2 dofs. 

#include <poplar/Graph.hpp>
#include <poplar/Engine.hpp>
#include <poplar/IPUModel.hpp>
#include <poplar/DeviceManager.hpp>
#include <poplin/MatMul.hpp>
#include <poplin/codelets.hpp>
#include <popops/codelets.hpp>
#include <poputil/TileMapping.hpp>
#include <popsys/CSRFunctions.hpp>
#include <popsys/codelets.hpp>

#include <chrono>

#include <iostream>
#include <fstream>
#include <string>

using namespace poplar;
using namespace poplar::program;
using namespace popsys;

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


// This function returns  compute set that updates the tensors beliefs_eta and beliefs_lambda
ComputeSet buildUpdateBeliefsCS(Graph &graph, 
                                unsigned n_varnodes,
                                unsigned var_dofs,
                                std::vector<unsigned> hnum_edges_pernode, 
                                Tensor var_nodes_dofs,
                                Tensor priors_eta, 
                                Tensor priors_lambda,
                                Tensor out_messages_eta,
                                Tensor out_messages_lambda,
                                Tensor beliefs_eta,
                                Tensor beliefs_lambda) {

  // A compute set is a set of vertices that are executed in parallel.
  ComputeSet cs_updatebeliefs = graph.addComputeSet("cs_updatebeliefs");

  // Adds vertices to the compute set. 
  for (unsigned varID = 0; varID < n_varnodes; ++varID) {
    VertexRef vtx_eta = graph.addVertex(cs_updatebeliefs, "UpdateBeliefEtaVertex");
    VertexRef vtx_lambda = graph.addVertex(cs_updatebeliefs, "UpdateBeliefLambdaVertex");

    // Set tile mapping of vertex.
    graph.setTileMapping(vtx_eta, varID);
    graph.setTileMapping(vtx_lambda, varID);

    // Connect data to the fields of the vertex. ie. define edges.
    graph.connect(vtx_eta["var_dofs"], var_nodes_dofs[varID]);
    graph.connect(vtx_lambda["var_dofs"], var_nodes_dofs[varID]);
    graph.connect(vtx_eta["prior_eta"], priors_eta.slice(varID * var_dofs, (varID + 1) * var_dofs));
    graph.connect(vtx_lambda["prior_lambda"], priors_lambda.slice(varID * var_dofs * var_dofs, (varID + 1) * var_dofs * var_dofs));

    // Number of edges in the grah connected to nodes with varID lower that the current varID
    unsigned edge_ix = 0;
    for (unsigned lowervarIDs = 0; lowervarIDs < varID; ++lowervarIDs) {
      edge_ix += hnum_edges_pernode[lowervarIDs];
    }

    graph.connect(vtx_eta["incoming_messages_eta"], out_messages_eta.slice(edge_ix * 2, (edge_ix + hnum_edges_pernode[varID]) * 2));
    graph.connect(vtx_lambda["incoming_messages_lambda"], out_messages_lambda.slice(edge_ix * 4, (edge_ix + hnum_edges_pernode[varID]) * 4));

    // Output fields of vertex
    graph.connect(vtx_eta["belief_eta"], beliefs_eta.slice(varID * var_dofs, (varID + 1) * var_dofs));
    graph.connect(vtx_lambda["belief_lambda"], beliefs_lambda.slice(varID * var_dofs * var_dofs, (varID + 1) * var_dofs * var_dofs));
  }
  return cs_updatebeliefs;
}


// This function returns  compute set that computes the outgoing messages at all factor nodes.
ComputeSet buildComputeMessagesCS(Graph &graph, 
                                unsigned n_varnodes,
                                unsigned n_edges,
                                unsigned var_dofs,
                                std::vector<unsigned> hnum_edges_pernode, 
                                std::vector<float> measurements_nodeIDs,
                                Tensor var_nodes_dofs,
                                Tensor factor_potentials_eta,
                                Tensor factor_potentials_lambda,
                                Tensor out_messages_eta,
                                Tensor out_messages_lambda,
                                Tensor pout_messages_eta,
                                Tensor pout_messages_lambda,
                                Tensor beliefs_eta,
                                Tensor beliefs_lambda) {

  // A compute set is a set of vertices that are executed in parallel.
  // This compute set contains the vertices to compute the outgoing messages at every factor node. 
  ComputeSet cs_computemessages = graph.addComputeSet("cs_computemessages");

  // Loop through factor nodes in the graph
  for (unsigned edgeID = 0; edgeID < n_edges; ++edgeID) {

    unsigned vnID0 = measurements_nodeIDs[2 * edgeID];
    unsigned vnID1 = measurements_nodeIDs[2 * edgeID + 1];

    unsigned tm = 0;
    // Choose tile mapping for edge. Strategy is to map half of each nodes edges onto the same tile.
    if (edgeID % 2 == 0) {
      tm = vnID0;
    }    
    else {
      tm = vnID1;
    }


    // Create vertices for the factor node. 2 vertices for computing eta messages in each direction and 2 for lambda messages
    VertexRef vx_eta0 = graph.addVertex(cs_computemessages, "ComputeMessageEtaVertex");  // varnode0 is outedge, varnode1 is nonoutedge
    VertexRef vx_lambda0 = graph.addVertex(cs_computemessages, "ComputeMessageLambdaVertex");

    VertexRef vx_eta1 = graph.addVertex(cs_computemessages, "ComputeMessageEtaVertex");  // varnode1 is outedge, varnode0 is nonoutedge
    VertexRef vx_lambda1 = graph.addVertex(cs_computemessages, "ComputeMessageLambdaVertex");

    // Prepare variables to connect to vertex fields
    unsigned varnodeID0 = measurements_nodeIDs[2 * edgeID];
    unsigned varnodeID1 = measurements_nodeIDs[2 * edgeID + 1];

    // Number of edges in the graph connected to nodes with varID lower than the varnodeID0
    unsigned edge_ix0 = 0;
    for (unsigned lowervarIDs = 0; lowervarIDs < varnodeID0; ++lowervarIDs) {
      edge_ix0 += hnum_edges_pernode[lowervarIDs];
    }
    unsigned edge_ix1 = 0;
    for (unsigned lowervarIDs = 0; lowervarIDs < varnodeID1; ++lowervarIDs) {
      edge_ix1 += hnum_edges_pernode[lowervarIDs];
    }

    // Index of factor node at the given variable node. 
    // Want to find how many times the variableID appears in measurements_nodeIDs before the edge we are at now. 
    unsigned node_factorix0 = 0;
    for (unsigned i = 0; i < edgeID * 2; ++i) {
      if (measurements_nodeIDs[i] == varnodeID0) {
        node_factorix0 += 1;
      }
    }
    unsigned node_factorix1 = 0;
    for (unsigned i = 0; i < edgeID * 2; ++i) {
      if (measurements_nodeIDs[i] == varnodeID1) {
        node_factorix1 += 1;
      }
    }


    // Connect vertex fields
    // Outedge is node 0
    graph.connect(vx_eta0["outedge_dofs"], var_nodes_dofs[varnodeID0]);
    graph.connect(vx_eta0["nonoutedge_dofs"], var_nodes_dofs[varnodeID1]);
    graph.connect(vx_eta0["f_outedge_eta_"], factor_potentials_eta.slice(edgeID * (var_dofs + var_dofs), edgeID * (var_dofs + var_dofs) + var_dofs));
    graph.connect(vx_eta0["f_nonoutedge_eta_"], factor_potentials_eta.slice(edgeID * (var_dofs + var_dofs) + var_dofs, (edgeID + 1) * (var_dofs + var_dofs)));
    graph.connect(vx_eta0["f_noe_noe_lambda_"], factor_potentials_lambda.slice(edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs) + 3 * var_dofs * var_dofs, (edgeID + 1) * (var_dofs + var_dofs) * (var_dofs + var_dofs)));
    graph.connect(vx_eta0["f_oe_noe_lambda_"], factor_potentials_lambda.slice(edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs) + var_dofs * var_dofs, edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs) + 2 * var_dofs * var_dofs));
    graph.connect(vx_eta0["belief_nonoutedge_eta_"], beliefs_eta.slice(varnodeID1 * var_dofs, (varnodeID1 + 1) * var_dofs));
    graph.connect(vx_eta0["belief_nonoutedge_lambda_"], beliefs_lambda.slice(varnodeID1 * var_dofs * var_dofs, (varnodeID1 + 1) * var_dofs * var_dofs));
    graph.connect(vx_eta0["pmess_nonoutedge_eta_"], pout_messages_eta.slice((edge_ix1 + node_factorix1) * 2, (edge_ix1 + node_factorix1 + 1) * 2)); // varnode1
    graph.connect(vx_eta0["pmess_nonoutedge_lambda_"], pout_messages_lambda.slice((edge_ix1 + node_factorix1) * 4, (edge_ix1 + node_factorix1 + 1) * 4)); // varnode1
    graph.connect(vx_eta0["pmess_outedge_eta_"], pout_messages_eta.slice((edge_ix0 + node_factorix0) * 2, (edge_ix0 + node_factorix0 + 1) * 2));  // varnode0
    graph.connect(vx_eta0["mess_outedge_eta_"], out_messages_eta.slice((edge_ix0 + node_factorix0) * 2, (edge_ix0 + node_factorix0 + 1) * 2));  // varnode0

    graph.connect(vx_lambda0["outedge_dofs"], var_nodes_dofs[varnodeID0]);
    graph.connect(vx_lambda0["nonoutedge_dofs"], var_nodes_dofs[varnodeID1]);
    graph.connect(vx_lambda0["f_oe_oe_lambda_"], factor_potentials_lambda.slice(edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs), edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs) + var_dofs * var_dofs));
    graph.connect(vx_lambda0["f_noe_noe_lambda_"], factor_potentials_lambda.slice(edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs) + 3 * var_dofs * var_dofs, (edgeID + 1) * (var_dofs + var_dofs) * (var_dofs + var_dofs)));
    graph.connect(vx_lambda0["f_oe_noe_lambda_"], factor_potentials_lambda.slice(edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs) + var_dofs * var_dofs, edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs) + 2 * var_dofs * var_dofs));
    graph.connect(vx_lambda0["f_noe_oe_lambda_"], factor_potentials_lambda.slice(edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs) + 2 * var_dofs * var_dofs, edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs) + 3 * var_dofs * var_dofs));
    graph.connect(vx_lambda0["belief_nonoutedge_lambda_"], beliefs_lambda.slice(varnodeID1 * var_dofs * var_dofs, (varnodeID1 + 1) * var_dofs * var_dofs));
    graph.connect(vx_lambda0["pmess_nonoutedge_lambda_"], pout_messages_lambda.slice((edge_ix1 + node_factorix1) * 4, (edge_ix1 + node_factorix1 + 1) * 4)); // varnode1
    graph.connect(vx_lambda0["pmess_outedge_lambda_"], pout_messages_lambda.slice((edge_ix0 + node_factorix0) * 4, (edge_ix0 + node_factorix0 + 1) * 4)); // varnode0
    graph.connect(vx_lambda0["mess_outedge_lambda_"], out_messages_lambda.slice((edge_ix0 + node_factorix0) * 4, (edge_ix0 + node_factorix0 + 1) * 4)); // varnode0

    // Message to node 1
    graph.connect(vx_eta1["outedge_dofs"], var_nodes_dofs[varnodeID1]);
    graph.connect(vx_eta1["nonoutedge_dofs"], var_nodes_dofs[varnodeID0]);
    graph.connect(vx_eta1["f_outedge_eta_"], factor_potentials_eta.slice(edgeID * (var_dofs + var_dofs) + var_dofs, (edgeID + 1) * (var_dofs + var_dofs)));
    graph.connect(vx_eta1["f_nonoutedge_eta_"], factor_potentials_eta.slice(edgeID * (var_dofs + var_dofs), edgeID * (var_dofs + var_dofs) + var_dofs));
    graph.connect(vx_eta1["f_noe_noe_lambda_"], factor_potentials_lambda.slice(edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs), edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs) + var_dofs * var_dofs));
    graph.connect(vx_eta1["f_oe_noe_lambda_"], factor_potentials_lambda.slice(edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs) + 2 * var_dofs * var_dofs, edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs) + 3 * var_dofs * var_dofs));
    graph.connect(vx_eta1["belief_nonoutedge_eta_"], beliefs_eta.slice(varnodeID0 * var_dofs, (varnodeID0 + 1) * var_dofs));
    graph.connect(vx_eta1["belief_nonoutedge_lambda_"], beliefs_lambda.slice(varnodeID0 * var_dofs * var_dofs, (varnodeID0 + 1) * var_dofs * var_dofs));
    graph.connect(vx_eta1["pmess_nonoutedge_eta_"], pout_messages_eta.slice((edge_ix0 + node_factorix0) * 2, (edge_ix0 + node_factorix0 + 1) * 2)); // varnode0
    graph.connect(vx_eta1["pmess_nonoutedge_lambda_"], pout_messages_lambda.slice((edge_ix0 + node_factorix0) * 4, (edge_ix0 + node_factorix0 + 1) * 4)); // varnode0
    graph.connect(vx_eta1["pmess_outedge_eta_"], pout_messages_eta.slice((edge_ix1 + node_factorix1) * 2, (edge_ix1 + node_factorix1 + 1) * 2));  // varnode1
    graph.connect(vx_eta1["mess_outedge_eta_"], out_messages_eta.slice((edge_ix1 + node_factorix1) * 2, (edge_ix1 + node_factorix1 + 1) * 2));  // varnode1

    graph.connect(vx_lambda1["outedge_dofs"], var_nodes_dofs[varnodeID1]);
    graph.connect(vx_lambda1["nonoutedge_dofs"], var_nodes_dofs[varnodeID0]);
    graph.connect(vx_lambda1["f_oe_oe_lambda_"], factor_potentials_lambda.slice(edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs) + 3 * var_dofs * var_dofs, (edgeID + 1) * (var_dofs + var_dofs) * (var_dofs + var_dofs)));
    graph.connect(vx_lambda1["f_noe_noe_lambda_"], factor_potentials_lambda.slice(edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs), edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs) + var_dofs * var_dofs));
    graph.connect(vx_lambda1["f_oe_noe_lambda_"], factor_potentials_lambda.slice(edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs) + 2 * var_dofs * var_dofs, edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs) + 3 * var_dofs * var_dofs));
    graph.connect(vx_lambda1["f_noe_oe_lambda_"], factor_potentials_lambda.slice(edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs) + var_dofs * var_dofs, edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs) + 2 * var_dofs * var_dofs));
    graph.connect(vx_lambda1["belief_nonoutedge_lambda_"], beliefs_lambda.slice(varnodeID0 * var_dofs * var_dofs, (varnodeID0 + 1) * var_dofs * var_dofs));
    graph.connect(vx_lambda1["pmess_nonoutedge_lambda_"], pout_messages_lambda.slice((edge_ix0 + node_factorix0) * 4, (edge_ix0 + node_factorix0 + 1) * 4)); // varnode0
    graph.connect(vx_lambda1["pmess_outedge_lambda_"], pout_messages_lambda.slice((edge_ix1 + node_factorix1) * 4, (edge_ix1 + node_factorix1 + 1) * 4)); // varnode1
    graph.connect(vx_lambda1["mess_outedge_lambda_"], out_messages_lambda.slice((edge_ix1 + node_factorix1) * 4, (edge_ix1 + node_factorix1 + 1) * 4)); // varnode1

    // Set tile mapping of vertex.
    // CHANGE THESE
    graph.setTileMapping(vx_eta0, n_varnodes + edgeID);
    graph.setTileMapping(vx_lambda0, n_varnodes + edgeID);
    graph.setTileMapping(vx_eta1, n_varnodes + edgeID);
    graph.setTileMapping(vx_lambda1, n_varnodes + edgeID);

  }
  return cs_computemessages;
}

// Compute a vector of the factor potentials from the measurements and associated sigmas on the measurements
// The measurement function is a simple 2D distance function
// vec<float> compute_factor_eta(vec measurements, vec sigmas){

//   return potentials_eta, potentials_lambda;
// }


int main() {

  // Load the data describing the properties of the graph. (These variables are not put into the graph but rather used to create it)
  unsigned n_edges = readUnsignedInt("../data/2d/n_edges.txt");  // total number of factor nodes / number of edges between variable nodes in the graph.
  unsigned n_varnodes = readUnsignedInt("../data/2d/n_varnodes.txt");
  unsigned var_dofs = readUnsignedInt("../data/2d/var_dofs.txt");

  // Data about observations required to set up vertices correctly.
  std::vector<float> measurements_nodeIDs = readFloatVector("../data/2d/measurements_nodeIDs.txt");

  // Load the data to be fed to the graph through the buffers
  std::vector<unsigned> hvar_nodes_dofs = readUnsignedIntVector("../data/2d/var_nodes_dofs.txt");
  std::vector<unsigned> hnum_edges_pernode = readUnsignedIntVector("../data/2d/num_edges_pernode.txt");
  std::vector<float> hpriors_eta = readFloatVector("../data/2d/priors_eta.txt");
  std::vector<float> hpriors_lambda = readFloatVector("../data/2d/priors_lambda.txt");
  std::vector<float> hmeasurements = readFloatVector("../data/2d/noisy_measurements.txt");
  std::vector<float> hmeas_variances = readFloatVector("../data/2d/meas_variances.txt");
  std::vector<float> hfactor_potentials_eta = readFloatVector("../data/2d/factor_potentials_eta.txt");
  std::vector<float> hfactor_potentials_lambda = readFloatVector("../data/2d/factor_potentials_lambda.txt");

  // Create buffers to hold beliefs when we stream them out
  float hbeliefs_eta [n_varnodes * var_dofs] = {};
  float hbeliefs_lambda [n_varnodes * var_dofs * var_dofs] = {};

  std::cout << "Number of variable nodes in the graph: " << n_varnodes << '\n';
  std::cout << "Number of edges in the graph: " << n_edges << '\n';
  std::cout << "Number of degrees of freedom at each variable node: " << var_dofs << '\n';

  assert (measurements_nodeIDs.size() == n_edges * 2);

  // Compute the factor potentials in advance from the measurements and the measurement sigma
  // hpotentials_eta, hpotentials_lambda = compute_factor_potentials(measurements, sigmas)


  // Create the IPU device
  std::cout << "\nAttaching to IPU device" << std::endl;
  IPUModel ipuModel;
  auto dm = DeviceManager::createDeviceManager();
  auto hwDevices = dm.getDevices(TargetType::IPU, 1);
  auto &device = hwDevices.front();
  device.attach();

  // Create the IPU model device
  // IPUModel ipuModel;
  // Device device = ipuModel.createDevice();

  // Create the Graph object 
  Graph graph(device.getTarget());
  // Add codelets for libraries and for my own vertices
  popops::addCodelets(graph);
  popsys::addCodelets(graph);
  graph.addCodelets("bp2d_codelets.cpp");


  // CREATE VARIABLE NODES

  Tensor var_nodes_dofs = graph.addVariable(UNSIGNED_INT, {n_varnodes}, "var_nodes_dofs"); // Number of dofs of variable stored at each variable node.
  Tensor num_edges_pernode = graph.addVariable(UNSIGNED_INT, {n_varnodes}, "num_edges_pernode");  // Number of other variable nodes that each variable node is connected to. 

  Tensor beliefs_eta = graph.addVariable(FLOAT, {var_dofs * n_varnodes}, "beliefs_eta");
  Tensor beliefs_lambda = graph.addVariable(FLOAT, {var_dofs * var_dofs * n_varnodes}, "beliefs_lambda");
  Tensor priors_eta = graph.addVariable(FLOAT, {var_dofs * n_varnodes}, "priors_eta");
  Tensor priors_lambda = graph.addVariable(FLOAT, {var_dofs * var_dofs * n_varnodes}, "priors_lambda");

  // Create data streams to write to graph variables
  DataStream instream_var_nodes_dofs = graph.addHostToDeviceFIFO("instream_var_nodes_dofs", UNSIGNED_INT, n_varnodes);
  DataStream instream_num_edges_pernode = graph.addHostToDeviceFIFO("instream_num_edges_pernode", UNSIGNED_INT, n_varnodes);

  // Create data streams to write into priors vector
  DataStream instream_priors_eta = graph.addHostToDeviceFIFO("instream_priors_eta", FLOAT, n_varnodes * var_dofs);
  DataStream instream_priors_lambda = graph.addHostToDeviceFIFO("instream_priors_lambda", FLOAT, n_varnodes * var_dofs * var_dofs);

  // Create data streams to read out beliefs    
  DataStream outstream_beliefs_eta = graph.addDeviceToHostFIFO("outstream_beliefs_eta", FLOAT, n_varnodes * var_dofs);
  DataStream outstream_beliefs_lambda = graph.addDeviceToHostFIFO("outstream_beliefs_lambda", FLOAT, n_varnodes * var_dofs * var_dofs);

  // Set tile mapping for variable nodes.
  for (unsigned varID = 0; varID < n_varnodes; ++varID) {
    // Set tile mapping for number of edges for a variable node.
    graph.setTileMapping(var_nodes_dofs[varID], varID);
    graph.setTileMapping(num_edges_pernode[varID], varID);

    // Set tile mapping for beliefs and priors
    graph.setTileMapping(beliefs_eta.slice(varID * var_dofs, (varID + 1) * var_dofs), varID);
    graph.setTileMapping(priors_eta.slice(varID * var_dofs, (varID + 1) * var_dofs), varID);
    graph.setTileMapping(beliefs_lambda.slice(varID * var_dofs * var_dofs, (varID + 1) * var_dofs * var_dofs), varID);
    graph.setTileMapping(priors_lambda.slice(varID * var_dofs * var_dofs, (varID + 1) * var_dofs * var_dofs), varID);
  }

  // CREATE FACTOR NODES FOR EACH EDGE

  // Can precompute factor potential using measurements z and sigma measurement
  Tensor measurements = graph.addVariable(FLOAT, {n_edges * 2}, "measurements");  // 2 is the dimensionality of the measurement at a single measurement node.
  Tensor meas_variances = graph.addVariable(FLOAT, {n_edges}, "meas_variances");

  // Factor potentials can be computed from the measurement and measurement variance.
  Tensor factor_potentials_eta = graph.addVariable(FLOAT, {n_edges * (var_dofs + var_dofs)}, "factor_potentials_eta");
  Tensor factor_potentials_lambda = graph.addVariable(FLOAT, {n_edges * (var_dofs + var_dofs) * (var_dofs + var_dofs)}, "factor_potentials_lambda");

  // Create tensors to store outwards messages at factor node
  Tensor out_messages_eta = graph.addVariable(FLOAT, {n_edges * 2 * 2}, "out_messages_eta"); // *2*2 as 2 directions to send messages each of which are 2d gaussians
  Tensor out_messages_lambda = graph.addVariable(FLOAT, {n_edges * 2 * 4}, "out_messages_lambda");
  Tensor pout_messages_eta = graph.addVariable(FLOAT, {n_edges * 2 * 2}, "out_messages_eta"); // *2*2 as 2 directions to send messages each of which are 2d gaussians
  Tensor pout_messages_lambda = graph.addVariable(FLOAT, {n_edges * 2 * 4}, "out_messages_lambda");

  // Create handle to write the measurements and measurement variances at the start. 
  DataStream instream_measurements = graph.addHostToDeviceFIFO("instream_measurements", FLOAT, n_edges * 2);
  DataStream instream_meas_variances = graph.addHostToDeviceFIFO("instream_meas_variances", FLOAT, n_edges);
  DataStream instream_factor_potentials_eta = graph.addHostToDeviceFIFO("instream_factor_potentials_eta", FLOAT, n_edges * (var_dofs + var_dofs));
  DataStream instream_factor_potentials_lambda = graph.addHostToDeviceFIFO("instream_factor_potentials_lambda", FLOAT, n_edges * (var_dofs + var_dofs) * (var_dofs + var_dofs));


  for (unsigned edgeID = 0; edgeID < n_edges; ++edgeID) {

    unsigned varnodeID0 = measurements_nodeIDs[2 * edgeID];
    unsigned varnodeID1 = measurements_nodeIDs[2 * edgeID + 1];

    unsigned tile = 0;
    // Choose tile mapping for edge. Strategy is to map half of each nodes edges onto the same tile.
    if (edgeID % 2 == 0) {
      tile = varnodeID0;
    }    
    else {
      tile = varnodeID1;
    }
    

    // First n_varnodes tiles are used for variable nodes, so use the next n_edges tile for factor nodes.
    graph.setTileMapping(measurements.slice(edgeID * 2, (edgeID + 1) * 2), n_varnodes + edgeID);
    graph.setTileMapping(meas_variances[edgeID], n_varnodes + edgeID);

    graph.setTileMapping(factor_potentials_eta.slice(edgeID * (var_dofs + var_dofs), (edgeID + 1) * (var_dofs + var_dofs)), n_varnodes + edgeID);
    graph.setTileMapping(factor_potentials_lambda.slice(edgeID * (var_dofs + var_dofs) * (var_dofs + var_dofs), (edgeID + 1) * (var_dofs + var_dofs) * (var_dofs + var_dofs)), n_varnodes + edgeID);
  

    // Number of edges in the graph connected to nodes with varID lower than the varnodeID0
    unsigned edge_ix0 = 0;
    for (unsigned lowervarIDs = 0; lowervarIDs < varnodeID0; ++lowervarIDs) {
      edge_ix0 += hnum_edges_pernode[lowervarIDs];
    }
    unsigned edge_ix1 = 0;
    for (unsigned lowervarIDs = 0; lowervarIDs < varnodeID1; ++lowervarIDs) {
      edge_ix1 += hnum_edges_pernode[lowervarIDs];
    }

    // Index of factor node at the given variable node. 
    // Want to find how many times the variableID appears in measurements_nodeIDs before the edge we are at now. 
    unsigned node_factorix0 = 0;
    for (unsigned i = 0; i < edgeID * 2; ++i) {
      if (measurements_nodeIDs[i] == varnodeID0) {
        node_factorix0 += 1;
      }
    }
    unsigned node_factorix1 = 0;
    for (unsigned i = 0; i < edgeID * 2; ++i) {
      if (measurements_nodeIDs[i] == varnodeID1) {
        node_factorix1 += 1;
      }
    }

    // Map tensor with outgoing message along both directions to factor tile.
    // Messages are ordered by node they are being sent to rather than by edgedID order.
    // Message going out to variable node listed first in measurement_nodeIDs
    graph.setTileMapping(out_messages_eta.slice((edge_ix0 + node_factorix0) * 2, (edge_ix0 + node_factorix0 + 1) * 2), n_varnodes + edgeID);
    graph.setTileMapping(out_messages_lambda.slice((edge_ix0 + node_factorix0) * 4, (edge_ix0 + node_factorix0 + 1) * 4), n_varnodes + edgeID);
    graph.setTileMapping(pout_messages_eta.slice((edge_ix0 + node_factorix0) * 2, (edge_ix0 + node_factorix0 + 1) * 2), n_varnodes + edgeID);
    graph.setTileMapping(pout_messages_lambda.slice((edge_ix0 + node_factorix0) * 4, (edge_ix0 + node_factorix0 + 1) * 4), n_varnodes + edgeID);
    // Message going out to variable node listed second in measurement_nodeIDs
    graph.setTileMapping(out_messages_eta.slice((edge_ix1 + node_factorix1) * 2, (edge_ix1 + node_factorix1 + 1) * 2), n_varnodes + edgeID);
    graph.setTileMapping(out_messages_lambda.slice((edge_ix1 + node_factorix1) * 4, (edge_ix1 + node_factorix1 + 1) * 4), n_varnodes + edgeID);
    graph.setTileMapping(pout_messages_eta.slice((edge_ix1 + node_factorix1) * 2, (edge_ix1 + node_factorix1 + 1) * 2), n_varnodes + edgeID);
    graph.setTileMapping(pout_messages_lambda.slice((edge_ix1 + node_factorix1) * 4, (edge_ix1 + node_factorix1 + 1) * 4), n_varnodes + edgeID);
    }


  // Build compute sets to update beliefs and compute factor messages
  auto cs_updatebeliefs = buildUpdateBeliefsCS(graph, n_varnodes, var_dofs, hnum_edges_pernode, var_nodes_dofs, priors_eta, priors_lambda,
                                                out_messages_eta, out_messages_lambda, beliefs_eta, beliefs_lambda);
  auto cs_computemessages = buildComputeMessagesCS(graph, n_varnodes, n_edges, var_dofs, hnum_edges_pernode, measurements_nodeIDs, var_nodes_dofs, 
                                          factor_potentials_eta,factor_potentials_lambda, out_messages_eta, out_messages_lambda, 
                                          pout_messages_eta, pout_messages_lambda, beliefs_eta, beliefs_lambda);


  // Create a control program 
  // Sequence prog;

  // Create a program to write initial data to tile
  Sequence wprog;
  wprog.add(Copy(instream_var_nodes_dofs, var_nodes_dofs));
  wprog.add(Copy(instream_num_edges_pernode, num_edges_pernode));

  // Stream priors to tiles
  wprog.add(Copy(instream_priors_eta, priors_eta));
  wprog.add(Copy(instream_priors_lambda, priors_lambda));

  // These actually aren't necessary in the linear case as we can precompute the factor potentials.
  wprog.add(Copy(instream_measurements, measurements));
  wprog.add(Copy(instream_meas_variances, meas_variances));

  // Write factor potentials ot tiles  
  wprog.add(Copy(instream_factor_potentials_eta, factor_potentials_eta));
  wprog.add(Copy(instream_factor_potentials_lambda, factor_potentials_lambda));

  // prog.add(wprog);

  // Print tensors that have been copied to the IPU
  // prog.add(PrintTensor("\nvar_nodes_dofs", var_nodes_dofs));
  // prog.add(PrintTensor("num_edges_pernode", num_edges_pernode));

  // prog.add(PrintTensor("\npriors_eta", priors_eta));
  // prog.add(PrintTensor("priors_lambda", priors_lambda));

  // prog.add(PrintTensor("beliefs_eta", beliefs_eta));
  // prog.add(PrintTensor("beliefs_lambda", beliefs_lambda));

  // prog.add(PrintTensor("\nfactor_potentials_eta", factor_potentials_eta));
  // prog.add(PrintTensor("factor_potentials_lambda", factor_potentials_lambda));

  // prog.add(PrintTensor("\nmeasurements", measurements));
  // prog.add(PrintTensor("meas_variances", meas_variances));

  // prog.add(PrintTensor("out_messages_eta", out_messages_eta));
  // prog.add(PrintTensor("out_messages_lambda", out_messages_lambda));

  Sequence comp_prog;

  // To throw errors when we have problems with floating point calculations
  FloatingPointBehaviour behaviour(true, true, true, false, true);  // inv, div0, oflo, esr, nanoo
  setFloatingPointBehaviour(graph, comp_prog, behaviour, "");

  comp_prog.add(Execute(cs_updatebeliefs));  // To send priors

  Sequence lprog;
  lprog.add(Execute(cs_computemessages));
  lprog.add(Execute(cs_updatebeliefs));
  lprog.add(Copy(out_messages_eta, pout_messages_eta)); // Messages from factor node have been used so can be copied to previous message holder
  lprog.add(Copy(out_messages_lambda, pout_messages_lambda));
  // lprog.add(PrintTensor("beliefs_eta", beliefs_eta));
  // lprog.add(PrintTensor("beliefs_lambda", beliefs_lambda));

  unsigned niters = 100;
  Repeat loop(niters, lprog);
  comp_prog.add(loop);

  // Program to stream out beliefs at the end from IPU
  Sequence rprog;
  rprog.add(Copy(beliefs_eta, outstream_beliefs_eta));
  rprog.add(Copy(beliefs_lambda, outstream_beliefs_lambda));


  // prog.add(PrintTensor("beliefs_eta", beliefs_eta));
  // prog.add(PrintTensor("beliefs_lambda", beliefs_lambda));

  // Create the engine
  Engine engine(graph, {wprog, comp_prog, rprog});
  engine.load(device);

  // Connect data streams for streaming in and out data
  engine.connectStream(instream_var_nodes_dofs, &hvar_nodes_dofs[0]);
  engine.connectStream(instream_num_edges_pernode, &hnum_edges_pernode[0]);
  engine.connectStream(instream_priors_eta, &hpriors_eta[0]);
  engine.connectStream(instream_priors_lambda, &hpriors_lambda[0]);
  engine.connectStream(instream_measurements, &hmeasurements[0]);
  engine.connectStream(instream_meas_variances, &hmeas_variances[0]);
  engine.connectStream(instream_factor_potentials_eta, &hfactor_potentials_eta[0]);
  engine.connectStream(instream_factor_potentials_lambda, &hfactor_potentials_lambda[0]);
  engine.connectStream(outstream_beliefs_eta, hbeliefs_eta);
  engine.connectStream(outstream_beliefs_lambda, hbeliefs_lambda);

  // Run programs
  std::cout << "Running program to stream initial data to IPU\n";
  engine.run(0);
  std::cout << "Initial data streaming complete\n\n";

  engine.resetExecutionProfile(); // Reset execution profile so that writing tensors doesn't appear on execution profile json

  std::cout << "Running message passing program\n";
  auto start_time = std::chrono::high_resolution_clock::now();   // Time how long the message passing program takes
  engine.run(1);
  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double, std::micro> elapsed = end_time - start_time;
  std::cout << "\nElapsed Time for message passing: " << elapsed.count() << " micro seconds" << "\n";
  std::cout << "Message passing program complete\n\n";


  // engine.run(2); // Program to stream out beliefs from IPU


  // // Print beliefs that are streamed out 
  // std::cout << "Eta beliefs: \n";
  // for (unsigned i = 0; i < n_varnodes * var_dofs; ++i) {
  //   std::cout << hbeliefs_eta[i] << " ";
  // }
  // std::cout << "\nLambda beliefs: \n";
  // for (unsigned i = 0; i < n_varnodes * var_dofs * var_dofs; ++i) {
  //   std::cout << hbeliefs_lambda[i] << " ";
  // }
  // std::cout << '\n';

  // Save intervals report
  std::ofstream fout("intervals.csv");
  engine.reportIntervals(fout);

  // Profiling
  char* log_dir = std::getenv("GC_PROFILE_LOG_DIR");
  std::string dir = log_dir ? std::string(log_dir) : ".";

  // Graph Report
  poplar::ProfileValue graphProfile = engine.getGraphProfile();
  std::ofstream graphReport;
  graphReport.open(dir + "/graph.json");
  poplar::serializeToJSON(graphReport, graphProfile);
  graphReport.close();

  // Execution Report
  poplar::ProfileValue execProfile = engine.getExecutionProfile();
  std::ofstream execReport;
  execReport.open(dir + "/execution.json");
  poplar::serializeToJSON(execReport, execProfile);
  execReport.close();

  return 0;
}
