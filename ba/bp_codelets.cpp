#include <poplar/Vertex.hpp>
#include <cmath>
// #include <print.h>

#include "matlib.cpp"
#include "bafuncs.cpp"

using namespace poplar;

// Global damping hyperparameters
float maxeta_damping = 0.4;
int num_undamped_iters = 8;
float dmu_threshold = 3e-3;
int min_linear_iters = 10;

float Nstds = 2.5;


// A vertex type to update eta of the belief of a variable node. 
class RelineariseFactorVertex : public Vertex {
public:
  Input<Vector<float>> measurement;
  Input<float> meas_variance;
  Input<Vector<float>> K_;
  Input<Vector<float>> kf_belief_eta_;
  Input<Vector<float>> kf_belief_lambda_;
  Input<Vector<float>> lmk_belief_eta_;
  Input<Vector<float>> lmk_belief_lambda_;

  InOut<Vector<float>> factor_eta_;
  InOut<Vector<float>> factor_lambda_cc_;
  InOut<Vector<float>> factor_lambda_ll_;
  InOut<Vector<float>> factor_lambda_cl_;
  InOut<Vector<float>> factor_lambda_lc_;

  Output<unsigned> robust_flag;

  bool compute() {

    // Create matrix objects
    Mat<float> meas(&measurement[0], 2, 1);
    Mat<float> K(&K_[0], 3, 3);
    Mat<float> kf_belief_eta(&kf_belief_eta_[0], 6, 1);
    Mat<float> kf_belief_lambda(&kf_belief_lambda_[0], 6, 6);
    Mat<float> lmk_belief_eta(&lmk_belief_eta_[0], 3, 1);
    Mat<float> lmk_belief_lambda(&lmk_belief_lambda_[0], 3, 3);
    Mat<float> factor_eta(&factor_eta_[0], 9, 1);
    Mat<float> factor_lambda_cc(&factor_lambda_cc_[0], 6, 6);
    Mat<float> factor_lambda_ll(&factor_lambda_ll_[0], 3, 3);
    Mat<float> factor_lambda_cl(&factor_lambda_cl_[0], 6, 3);
    Mat<float> factor_lambda_lc(&factor_lambda_lc_[0], 3, 6);

    // Zero memory first
    for (unsigned i = 0; i < factor_eta.getnRows(); ++i) {
        factor_eta(i,0) = 0;
    } 
    for (unsigned i = 0; i < factor_lambda_cc.getnRows(); ++i) {
        for (unsigned j = 0; j < factor_lambda_cc.getnCols(); ++j) {
            factor_lambda_cc(i,j) = 0;
        }
    } 
    for (unsigned i = 0; i < factor_lambda_ll.getnRows(); ++i) {
        for (unsigned j = 0; j < factor_lambda_ll.getnCols(); ++j) {
            factor_lambda_ll(i,j) = 0;
        }
    } 
    for (unsigned i = 0; i < factor_lambda_cl.getnRows(); ++i) {
        for (unsigned j = 0; j < factor_lambda_cl.getnCols(); ++j) {
            factor_lambda_cl(i,j) = 0;
        }
    }
    for (unsigned i = 0; i < factor_lambda_lc.getnRows(); ++i) {
        for (unsigned j = 0; j < factor_lambda_lc.getnCols(); ++j) {
            factor_lambda_lc(i,j) = 0;
        }
    } 

    // Get linearisation point as x0 of belief
    float kf_belief_x0_ [6] = {}; 
    float kf_belief_sigma_ [36] = {}; 
    float lmk_belief_x0_ [3] = {}; 
    float lmk_belief_sigma_ [9] = {}; 
    Mat<float> kf_belief_x0(kf_belief_x0_, 6, 1);
    Mat<float> kf_belief_sigma(kf_belief_sigma_, 6, 6);
    Mat<float> lmk_belief_x0(lmk_belief_x0_, 3, 1);
    Mat<float> lmk_belief_sigma(lmk_belief_sigma_, 3, 3);
    inf2mean6x6(kf_belief_eta, kf_belief_lambda, kf_belief_x0, kf_belief_sigma);
    inf2mean3x3(lmk_belief_eta, lmk_belief_lambda, lmk_belief_x0, lmk_belief_sigma);

    float J_kf_ [12] = {};
    float J_lmk_ [6] = {};
    Mat<float> J_kf(J_kf_, 2, 6);
    Mat<float> J_lmk(J_lmk_, 2, 3);

    Jac(kf_belief_x0, lmk_belief_x0, K, J_kf, J_lmk);//, J_proj, y_cf, yhomog, Tw2c, angles);

    matMul(J_kf, J_kf, factor_lambda_cc, true, false);
    matMul(J_lmk, J_lmk, factor_lambda_ll, true, false);
    matMul(J_kf, J_lmk, factor_lambda_cl, true, false);

    // Calculate predicted coordinates of the measurement given the beliefs
    float hx0_ [2] = {};
    Mat<float> hx0(hx0_, 2, 1);
    hfunc(kf_belief_x0, lmk_belief_x0, K, hx0);

    float buffer6 [2] = {};
    float buffer7 [9] = {};
    Mat<float> eta_buffer(buffer6, 2, 1);
    Mat<float> x0(buffer7, 9, 1);
    for (unsigned i = 0; i < 6; ++i) {
        x0(i,0) = kf_belief_x0(i,0);
    }
    for (unsigned i = 0; i < 3; ++i) {
        x0(i+6,0) = lmk_belief_x0(i,0);
    }

    float J_ [18] = {};
    Mat<float> J(J_, 2, 9);
    for (unsigned i = 0; i < J_kf.getnRows(); ++i) {
        for (unsigned j = 0; j < J_kf.getnCols(); ++j) {
            J(i,j) = J_kf(i,j);
        }
    } 
    for (unsigned i = 0; i < J_lmk.getnRows(); ++i) {
        for (unsigned j = 0; j < J_lmk.getnCols(); ++j) {
            J(i,j+6) = J_lmk(i,j);
        }
    } 
    matMul(J, x0, eta_buffer);
    matSum(eta_buffer, meas, eta_buffer);
    matSum(eta_buffer, hx0, eta_buffer, true);  // true x0s subtract
    matMul(J, eta_buffer, factor_eta, true, false);

    // Calculate Huber loss function
    float err = std::sqrt( (hx0(0,0) - meas(0,0))*(hx0(0,0) - meas(0,0)) + (hx0(1,0) - meas(1,0))*(hx0(1,0) - meas(1,0)) );
    float modified_meas_var = meas_variance;
    if (err > Nstds * std::sqrt(meas_variance)) {
      *robust_flag = 1;
      modified_meas_var = meas_variance * err * err / (2 * (Nstds * std::sqrt(meas_variance) * err - 0.5 * Nstds * Nstds * meas_variance));
    }
    else { *robust_flag = 0;}

    for (unsigned i = 0; i < factor_lambda_cc.getnRows(); ++i) {
        for (unsigned j = 0; j < factor_lambda_cc.getnCols(); ++j) {
            factor_lambda_cc(i,j) /= modified_meas_var;
        }
    } 
    for (unsigned i = 0; i < factor_lambda_ll.getnRows(); ++i) {
        for (unsigned j = 0; j < factor_lambda_ll.getnCols(); ++j) {
            factor_lambda_ll(i,j) /= modified_meas_var;
        }
    } 
    for (unsigned i = 0; i < factor_lambda_cl.getnRows(); ++i) {
        for (unsigned j = 0; j < factor_lambda_cl.getnCols(); ++j) {
            factor_lambda_cl(i,j) /= modified_meas_var;
        }
    }
    for (unsigned i = 0; i < factor_lambda_lc.getnRows(); ++i) {
        for (unsigned j = 0; j < factor_lambda_lc.getnCols(); ++j) {
            factor_lambda_lc(i,j) = factor_lambda_cl(j,i);
        }
    } 

    for (unsigned i = 0; i < factor_eta.getnRows(); ++i) {
        for (unsigned j = 0; j < factor_eta.getnCols(); ++j) {
            factor_eta(i,j) /= modified_meas_var;
        }
    } 

    return true;
  }
};


// A vertex type to weaken the prior at a variable node by a given scaling 
class WeakenPriorVertex : public Vertex {
public:
  Input<float> scaling;
  InOut<unsigned> weaken_flag;

  InOut<Vector<float>> prior_eta;
  InOut<Vector<float>> prior_lambda;

  bool compute() {
    if ((weaken_flag==5) || (weaken_flag==4) || (weaken_flag==3) || (weaken_flag==2) || (weaken_flag==1)) {
      weaken_flag -= 1;
      for (unsigned i = 0; i < prior_eta.size(); ++i) {
        prior_eta[i] *= scaling;
      }

      for (unsigned i = 0; i < prior_lambda.size(); ++i) {
        prior_lambda[i] *= scaling;
      }
    }
    return true;
  }
};


// // Only works when run on IPU model
// class PrintVertex : public Vertex {
// public:
//   Input<Vector<float>> vec;

//   bool compute() {

//     for (unsigned i = 0; i < vec.size(); ++i) {
//       printf("Vector is: %.10f \n", *(&vec[i]));
//     }

//     return true;
//   }
// };

class PrepMessageVertex : public Vertex {
// Select damping factor for node and relinearise if necessary. 
public:

  InOut<float> damping;
  InOut<int> damping_count;
  Input<unsigned> active_flag;
  Output<unsigned> robust_flag;
  
  Input<Vector<float>> measurement;
  Input<Vector<float>> K_;
  Input<float> meas_variance;
  Input<Vector<float>> kf_belief_eta_;
  Input<Vector<float>> kf_belief_lambda_;
  Input<Vector<float>> lmk_belief_eta_;
  Input<Vector<float>> lmk_belief_lambda_;

  Input<Vector<float>> oldmu;
  InOut<Vector<float>> mu;
  InOut<float> dmu;
  InOut<Vector<float>> factor_eta_;
  InOut<Vector<float>> factor_lambda_cc_;
  InOut<Vector<float>> factor_lambda_ll_;
  InOut<Vector<float>> factor_lambda_cl_;
  InOut<Vector<float>> factor_lambda_lc_;

  bool compute() {
    if (active_flag == 1) {      

      // Set damping factor 
      if (0 == damping_count) {
        *damping = maxeta_damping;
      }
      *damping_count += 1;

      Mat<float> kf_belief_eta(&kf_belief_eta_[0], 6, 1);
      Mat<float> kf_belief_lambda(&kf_belief_lambda_[0], 6, 6);
      Mat<float> lmk_belief_eta(&lmk_belief_eta_[0], 3, 1);
      Mat<float> lmk_belief_lambda(&lmk_belief_lambda_[0], 3, 3);

      // Get linearisation point as x0 of belief
      float kf_belief_x0_ [6] = {}; 
      float kf_belief_sigma_ [36] = {}; 
      float lmk_belief_x0_ [3] = {}; 
      float lmk_belief_sigma_ [9] = {}; 
      Mat<float> kf_belief_x0(kf_belief_x0_, 6, 1);
      Mat<float> kf_belief_sigma(kf_belief_sigma_, 6, 6);
      Mat<float> lmk_belief_x0(lmk_belief_x0_, 3, 1);
      Mat<float> lmk_belief_sigma(lmk_belief_sigma_, 3, 3);
      inf2mean6x6(kf_belief_eta, kf_belief_lambda, kf_belief_x0, kf_belief_sigma);
      inf2mean3x3(lmk_belief_eta, lmk_belief_lambda, lmk_belief_x0, lmk_belief_sigma);

      // Update mu
      *dmu = 0.0;
      for (unsigned i = 0; i < 6; ++i) {
        *dmu += (oldmu[i] - kf_belief_x0(i,0)) * (oldmu[i] - kf_belief_x0(i,0));
        mu[i] = kf_belief_x0(i,0);
      }
      for (unsigned i = 0; i < 3; ++i) {
        *dmu += (oldmu[i+6] - lmk_belief_x0(i,0)) * (oldmu[i+6] - lmk_belief_x0(i,0));
        mu[i+6] = lmk_belief_x0(i,0);
      }
      *dmu = std::sqrt(dmu);

      // Relinearise if necessary
      if ((*dmu < dmu_threshold) && (*damping_count > min_linear_iters - num_undamped_iters)) {

        *damping = 0.0;
        *damping_count = - num_undamped_iters;

        // Create matrix objects
        Mat<float> meas(&measurement[0], 2, 1);
        Mat<float> K(&K_[0], 3, 3);
        Mat<float> factor_eta(&factor_eta_[0], 9, 1);
        Mat<float> factor_lambda_cc(&factor_lambda_cc_[0], 6, 6);
        Mat<float> factor_lambda_ll(&factor_lambda_ll_[0], 3, 3);
        Mat<float> factor_lambda_cl(&factor_lambda_cl_[0], 6, 3);
        Mat<float> factor_lambda_lc(&factor_lambda_lc_[0], 3, 6);

        float J_kf_ [12] = {};
        float J_lmk_ [6] = {};
        Mat<float> J_kf(J_kf_, 2, 6);
        Mat<float> J_lmk(J_lmk_, 2, 3);

        Jac(kf_belief_x0, lmk_belief_x0, K, J_kf, J_lmk);//, J_proj, y_cf, yhomog, Tw2c, angles);

        matMul(J_kf, J_kf, factor_lambda_cc, true, false);
        matMul(J_lmk, J_lmk, factor_lambda_ll, true, false);
        matMul(J_kf, J_lmk, factor_lambda_cl, true, false);

        // Calculate predicted coordinates of the measurement given the beliefs
        float hx0_ [2] = {};
        Mat<float> hx0(hx0_, 2, 1);
        hfunc(kf_belief_x0, lmk_belief_x0, K, hx0);

        float buffer6 [2] = {};
        float buffer7 [9] = {};
        Mat<float> eta_buffer(buffer6, 2, 1);
        Mat<float> x0(buffer7, 9, 1);
        for (unsigned i = 0; i < 6; ++i) {
            x0(i,0) = kf_belief_x0(i,0);
        }
        for (unsigned i = 0; i < 3; ++i) {
            x0(i+6,0) = lmk_belief_x0(i,0);
        }

        float J_ [18] = {};
        Mat<float> J(J_, 2, 9);
        for (unsigned i = 0; i < J_kf.getnRows(); ++i) {
            for (unsigned j = 0; j < J_kf.getnCols(); ++j) {
                J(i,j) = J_kf(i,j);
            }
        } 
        for (unsigned i = 0; i < J_lmk.getnRows(); ++i) {
            for (unsigned j = 0; j < J_lmk.getnCols(); ++j) {
                J(i,j+6) = J_lmk(i,j);
            }
        } 
        matMul(J, x0, eta_buffer);
        matSum(eta_buffer, meas, eta_buffer);
        matSum(eta_buffer, hx0, eta_buffer, true);  // true x0s subtract
        matMul(J, eta_buffer, factor_eta, true, false);



      // Calculate Huber loss function
      float err = std::sqrt( (hx0(0,0) - meas(0,0))*(hx0(0,0) - meas(0,0)) + (hx0(1,0) - meas(1,0))*(hx0(1,0) - meas(1,0)) );
      float modified_meas_var = meas_variance;
      if (err > Nstds * std::sqrt(meas_variance)) {
        *robust_flag = 1;
        modified_meas_var = meas_variance * err * err / (2 * (Nstds * std::sqrt(meas_variance) * err - 0.5 * Nstds * Nstds * meas_variance));
      }
      else { *robust_flag = 0;}
      for (unsigned i = 0; i < factor_lambda_cc.getnRows(); ++i) {
          for (unsigned j = 0; j < factor_lambda_cc.getnCols(); ++j) {
              factor_lambda_cc(i,j) /= modified_meas_var;
          }
      } 
      for (unsigned i = 0; i < factor_lambda_ll.getnRows(); ++i) {
          for (unsigned j = 0; j < factor_lambda_ll.getnCols(); ++j) {
              factor_lambda_ll(i,j) /= modified_meas_var;
          }
      } 
      for (unsigned i = 0; i < factor_lambda_cl.getnRows(); ++i) {
          for (unsigned j = 0; j < factor_lambda_cl.getnCols(); ++j) {
              factor_lambda_cl(i,j) /= modified_meas_var;
          }
      }
      for (unsigned i = 0; i < factor_lambda_lc.getnRows(); ++i) {
          for (unsigned j = 0; j < factor_lambda_lc.getnCols(); ++j) {
              factor_lambda_lc(i,j) = factor_lambda_cl(j,i);
          }
      } 

      for (unsigned i = 0; i < factor_eta.getnRows(); ++i) {
          for (unsigned j = 0; j < factor_eta.getnCols(); ++j) {
              factor_eta(i,j) /= modified_meas_var;
          }
      } 
      }
    }

    return true;
  }
};

// A vertex type to compute the eta message going to the node variables first in the factor potential
class ComputeCamMessageEtaVertex : public Vertex {
public:
  // The inputs read a vector of values (i.e. an ordered, contiguous set of values in memory) from the graph.
  // Must input beliefs separately as the beliefs vector is ordered in order of variableID
  Input<float> damping; 
  Input<unsigned> active_flag; 

  Input<unsigned> outedge_dofs; 
  Input<unsigned> nonoutedge_dofs; 

  Input<Vector<float>> f_outedge_eta_;  // Underscore at the end if to differentiate from mat object
  Input<Vector<float>> f_nonoutedge_eta_;
  Input<Vector<float>> f_noe_noe_lambda_;
  Input<Vector<float>> f_oe_noe_lambda_;

  Input<Vector<float>> belief_nonoutedge_eta_;
  Input<Vector<float>> belief_nonoutedge_lambda_;

  // Vectors containing previous outward messages from the factor node
  // Must input them as separate vectors as the out_messages vector is ordered by variable node it is sent to rather than by factor node.
  // For computing message  
  Input<Vector<float>> pmess_nonoutedge_eta_;  
  Input<Vector<float>> pmess_nonoutedge_lambda_;  
  // For message damping
  Input<Vector<float>> pmess_outedge_eta_;  

  // The ouput is to a vector in the graph which is the new outgoing messages from the factor node. 
  InOut<Vector<float>> mess_outedge_eta_;  

  bool compute() {

    if (active_flag == 1) {

      // Create input matrix objects
      Mat<float> f_outedge_eta(&f_outedge_eta_[0], outedge_dofs, 1); 
      Mat<float> f_nonoutedge_eta(&f_nonoutedge_eta_[0], nonoutedge_dofs, 1);
      Mat<float> f_noe_noe_lambda(&f_noe_noe_lambda_[0], nonoutedge_dofs, nonoutedge_dofs);
      Mat<float> f_oe_noe_lambda(&f_oe_noe_lambda_[0], outedge_dofs, nonoutedge_dofs);

      Mat<float> belief_nonoutedge_eta(&belief_nonoutedge_eta_[0], nonoutedge_dofs, 1);
      Mat<float> belief_nonoutedge_lambda(&belief_nonoutedge_lambda_[0], nonoutedge_dofs, nonoutedge_dofs);  

      Mat<float> pmess_nonoutedge_eta(&pmess_nonoutedge_eta_[0], nonoutedge_dofs, 1);
      Mat<float> pmess_nonoutedge_lambda(&pmess_nonoutedge_lambda_[0], nonoutedge_dofs, nonoutedge_dofs);
      Mat<float> pmess_outedge_eta(&pmess_outedge_eta_[0], outedge_dofs, 1);

      // Create output matrix object
      Mat<float> mess_outedge_eta(&mess_outedge_eta_[0], outedge_dofs, 1);

      // Create buffers to hold intermediate data in calculation
      float buffer1 [3*3] = {}; 
      float buffer2 [3*3] = {}; 
      float buffer3 [6*3] = {}; 
      float buffer4 [3] = {}; 
      float buffer5 [6] = {}; 
      float buffer6 [6] = {};
      Mat<float> lambda_noe_noe_dash(buffer1, 3, 3); 
      Mat<float> lambda_noe_noe_dashinv(buffer2, 3, 3);
      Mat<float> lambdaprod(buffer3, 6, 3);
      Mat<float> eta_noe_dash(buffer4, 3, 1);
      Mat<float> eta_noe_sum(buffer5, 6, 1);
      Mat<float> hmess_outedge_eta(buffer6, 6, 1);

      // Calculate message eta by marginalisation
      matSum(f_noe_noe_lambda, belief_nonoutedge_lambda, lambda_noe_noe_dash);  // Sum block of fp lambdamatrix and belief lambda
      matSum(lambda_noe_noe_dash, pmess_nonoutedge_lambda, lambda_noe_noe_dash, true);  // Subtract from above the previous message lambda

      inv3x3(lambda_noe_noe_dash, lambda_noe_noe_dashinv);
      matMul(f_oe_noe_lambda, lambda_noe_noe_dashinv, lambdaprod);

      matSum(f_nonoutedge_eta, belief_nonoutedge_eta, eta_noe_dash);  // Sum block of fp lambdamatrix and belief lambda
      matSum(eta_noe_dash, pmess_nonoutedge_eta, eta_noe_dash, true);  // Subtract from above the previous message lambda
     
      matMul(lambdaprod, eta_noe_dash, eta_noe_sum);
      matSum(f_outedge_eta, eta_noe_sum, hmess_outedge_eta, true);

      for (unsigned i = 0; i < hmess_outedge_eta.getnRows(); ++i) {
        for (unsigned j = 0; j < hmess_outedge_eta.getnCols(); ++j) {
          mess_outedge_eta(i,j) = hmess_outedge_eta(i,j) * (1 - damping) + pmess_outedge_eta(i,j) * damping;
        }
      } 
    }
    else {
      for (unsigned i = 0; i < mess_outedge_eta_.size(); ++i) {
        mess_outedge_eta_[i] = 0.0;
      }
    } 

    return true;
  }
};

// A vertex type to compute the eta message going to the node variables first in the factor potential
class ComputeLmkMessageEtaVertex : public Vertex {
public:
  // The inputs read a vector of values (i.e. an ordered, contiguous set of values in memory) from the graph.
  // Must input beliefs separately as the beliefs vector is ordered in order of variableID
  Input<float> damping;  
  Input<unsigned> outedge_dofs; 
  Input<unsigned> nonoutedge_dofs; 
  Input<unsigned> active_flag;

  Input<Vector<float>> f_outedge_eta_;  // Underscore at the end if to differentiate from mat object
  Input<Vector<float>> f_nonoutedge_eta_;
  Input<Vector<float>> f_noe_noe_lambda_;
  Input<Vector<float>> f_oe_noe_lambda_;

  Input<Vector<float>> belief_nonoutedge_eta_;
  Input<Vector<float>> belief_nonoutedge_lambda_;

  // Vectors containing previous outward messages from the factor node
  // Must input them as separate vectors as the out_messages vector is ordered by variable node it is sent to rather than by factor node.
  // For computing message  
  Input<Vector<float>> pmess_nonoutedge_eta_;  
  Input<Vector<float>> pmess_nonoutedge_lambda_;  
  // For message damping
  Input<Vector<float>> pmess_outedge_eta_;  

  // The ouput is to a vector in the graph which is the new outgoing messages from the factor node. 
  InOut<Vector<float>> mess_outedge_eta_;  

  bool compute() {
    if (active_flag == 1) {
      // Create input matrix objects
      Mat<float> f_outedge_eta(&f_outedge_eta_[0], outedge_dofs, 1); 
      Mat<float> f_nonoutedge_eta(&f_nonoutedge_eta_[0], nonoutedge_dofs, 1);
      Mat<float> f_noe_noe_lambda(&f_noe_noe_lambda_[0], nonoutedge_dofs, nonoutedge_dofs);
      Mat<float> f_oe_noe_lambda(&f_oe_noe_lambda_[0], outedge_dofs, nonoutedge_dofs);

      Mat<float> belief_nonoutedge_eta(&belief_nonoutedge_eta_[0], nonoutedge_dofs, 1);
      Mat<float> belief_nonoutedge_lambda(&belief_nonoutedge_lambda_[0], nonoutedge_dofs, nonoutedge_dofs);  

      Mat<float> pmess_nonoutedge_eta(&pmess_nonoutedge_eta_[0], nonoutedge_dofs, 1);
      Mat<float> pmess_nonoutedge_lambda(&pmess_nonoutedge_lambda_[0], nonoutedge_dofs, nonoutedge_dofs);
      Mat<float> pmess_outedge_eta(&pmess_outedge_eta_[0], outedge_dofs, 1);

      // Create output matrix object
      Mat<float> mess_outedge_eta(&mess_outedge_eta_[0], outedge_dofs, 1);

      // Create buffers to hold intermediate data in calculation
      float buffer1 [6*6] = {}; 
      float buffer2 [6*6] = {}; 
      float buffer3 [3*6] = {}; 
      float buffer4 [6] = {}; 
      float buffer5 [3] = {}; 
      float buffer6 [3] = {}; 
      Mat<float> lambda_noe_noe_dash(buffer1, 6, 6); 
      Mat<float> lambda_noe_noe_dashinv(buffer2, 6, 6);
      Mat<float> lambdaprod(buffer3, 3, 6);
      Mat<float> eta_noe_dash(buffer4, 6, 1);
      Mat<float> eta_noe_sum(buffer5, 3, 1);
      Mat<float> hmess_outedge_eta(buffer6, 3, 1);

      // Calculate message eta by marginalisation
      matSum(f_noe_noe_lambda, belief_nonoutedge_lambda, lambda_noe_noe_dash);  // Sum block of fp lambdamatrix and belief lambda
      matSum(lambda_noe_noe_dash, pmess_nonoutedge_lambda, lambda_noe_noe_dash, true);  // Subtract from above the previous message lambda

      inv6x6(lambda_noe_noe_dash, lambda_noe_noe_dashinv);
      matMul(f_oe_noe_lambda, lambda_noe_noe_dashinv, lambdaprod);

      matSum(f_nonoutedge_eta, belief_nonoutedge_eta, eta_noe_dash);  // Sum block of fp lambdamatrix and belief lambda
      matSum(eta_noe_dash, pmess_nonoutedge_eta, eta_noe_dash, true);  // Subtract from above the previous message lambda
      
      matMul(lambdaprod, eta_noe_dash, eta_noe_sum);
      matSum(f_outedge_eta, eta_noe_sum, hmess_outedge_eta, true);

      for (unsigned i = 0; i < hmess_outedge_eta.getnRows(); ++i) {
        for (unsigned j = 0; j < hmess_outedge_eta.getnCols(); ++j) {
          mess_outedge_eta(i,j) = hmess_outedge_eta(i,j) * (1 - damping) + pmess_outedge_eta(i,j) * damping;
        }
      } 

    }
    else {
      for (unsigned i = 0; i < mess_outedge_eta_.size(); ++i) {
        mess_outedge_eta_[i] = 0.0;
      }
    } 

    return true;
  }
};


// A vertex type to compute the messages from a factor node. 
class ComputeCamMessageLambdaVertex : public Vertex {
public:
  // The inputs read a vector of values (i.e. an ordered, contiguous set of values in memory) from the graph.
  // Must input beliefs separately as the beliefs vector is ordered in order of variableID
  Input<unsigned> outedge_dofs; 
  Input<unsigned> nonoutedge_dofs; 
  Input<unsigned> active_flag;

  Input<Vector<float>> f_oe_oe_lambda_;
  Input<Vector<float>> f_noe_noe_lambda_;
  Input<Vector<float>> f_oe_noe_lambda_;
  Input<Vector<float>> f_noe_oe_lambda_;

  Input<Vector<float>> belief_nonoutedge_lambda_;

  // Vectors containing previous outward messages from the factor node
  // Must input them as separate vectors as the out_messages vector is ordered by variable node it is sent to rather than by factor node.
  // For computing message  
  Input<Vector<float>> pmess_nonoutedge_lambda_;  



  // The ouput is to a vector in the graph which is the new outgoing messages from the factor node. 
  InOut<Vector<float>> mess_outedge_lambda_;  

  bool compute() {
    if (active_flag == 1) {

      // Create input matrix objects
      Mat<float> f_oe_oe_lambda(&f_oe_oe_lambda_[0], outedge_dofs, outedge_dofs); 
      Mat<float> f_noe_noe_lambda(&f_noe_noe_lambda_[0], nonoutedge_dofs, nonoutedge_dofs);
      Mat<float> f_oe_noe_lambda(&f_oe_noe_lambda_[0], outedge_dofs, nonoutedge_dofs);
      Mat<float> f_noe_oe_lambda(&f_noe_oe_lambda_[0], nonoutedge_dofs, outedge_dofs);

      Mat<float> belief_nonoutedge_lambda(&belief_nonoutedge_lambda_[0], nonoutedge_dofs, nonoutedge_dofs);  

      Mat<float> pmess_nonoutedge_lambda(&pmess_nonoutedge_lambda_[0], nonoutedge_dofs, nonoutedge_dofs);

      // Create output matrix object
      Mat<float> mess_outedge_lambda(&mess_outedge_lambda_[0], outedge_dofs, outedge_dofs);

      // Create buffers to hold intermediate data in calculation
      float buffer1 [3*3] = {}; 
      float buffer2 [3*3] = {}; 
      float buffer3 [6*3] = {}; 
      float buffer4 [6*6] = {}; 
      Mat<float> lambda_noe_noe_dash(buffer1, 3, 3); 
      Mat<float> lambda_noe_noe_dashinv(buffer2, 3, 3);
      Mat<float> lambdaprod(buffer3, 6, 3);
      Mat<float> lambda_noe_noe_sum(buffer4, 6, 6);

      // Calculate message lambda by marginalisation
      matSum(f_noe_noe_lambda, belief_nonoutedge_lambda, lambda_noe_noe_dash);  // Sum block of fp lambdamatrix and belief lambda
      matSum(lambda_noe_noe_dash, pmess_nonoutedge_lambda, lambda_noe_noe_dash, true);  // Subtract from above the previous message lambda

      inv3x3(lambda_noe_noe_dash, lambda_noe_noe_dashinv);
      matMul(f_oe_noe_lambda, lambda_noe_noe_dashinv, lambdaprod);

      matMul(lambdaprod, f_noe_oe_lambda, lambda_noe_noe_sum);

      matSum(f_oe_oe_lambda, lambda_noe_noe_sum, mess_outedge_lambda, true);  

    }
    else {
      for (unsigned i = 0; i < mess_outedge_lambda_.size(); ++i) {
        mess_outedge_lambda_[i] = 0.0;
      }
    } 

    return true;
  }
};

// A vertex type to compute the messages from a factor node. 
class ComputeLmkMessageLambdaVertex : public Vertex {
public:
  // The inputs read a vector of values (i.e. an ordered, contiguous set of values in memory) from the graph.
  // Must input beliefs separately as the beliefs vector is ordered in order of variableID
  Input<unsigned> outedge_dofs; 
  Input<unsigned> nonoutedge_dofs; 
  Input<unsigned> active_flag;

  Input<Vector<float>> f_oe_oe_lambda_;
  Input<Vector<float>> f_noe_noe_lambda_;
  Input<Vector<float>> f_oe_noe_lambda_;
  Input<Vector<float>> f_noe_oe_lambda_;

  Input<Vector<float>> belief_nonoutedge_lambda_;

  // Vectors containing previous outward messages from the factor node
  // Must input them as separate vectors as the out_messages vector is ordered by variable node it is sent to rather than by factor node.
  // For computing message  
  Input<Vector<float>> pmess_nonoutedge_lambda_;  

  // The ouput is to a vector in the graph which is the new outgoing messages from the factor node. 
  InOut<Vector<float>> mess_outedge_lambda_;  

  bool compute() {
    if (active_flag == 1) {

      // Create input matrix objects
      Mat<float> f_oe_oe_lambda(&f_oe_oe_lambda_[0], outedge_dofs, outedge_dofs); 
      Mat<float> f_noe_noe_lambda(&f_noe_noe_lambda_[0], nonoutedge_dofs, nonoutedge_dofs);
      Mat<float> f_oe_noe_lambda(&f_oe_noe_lambda_[0], outedge_dofs, nonoutedge_dofs);
      Mat<float> f_noe_oe_lambda(&f_noe_oe_lambda_[0], nonoutedge_dofs, outedge_dofs);

      Mat<float> belief_nonoutedge_lambda(&belief_nonoutedge_lambda_[0], nonoutedge_dofs, nonoutedge_dofs);  

      Mat<float> pmess_nonoutedge_lambda(&pmess_nonoutedge_lambda_[0], nonoutedge_dofs, nonoutedge_dofs);

      // Create output matrix object
      Mat<float> mess_outedge_lambda(&mess_outedge_lambda_[0], outedge_dofs, outedge_dofs);

      // Create buffers to hold intermediate data in calculation
      float buffer1 [6*6] = {}; 
      float buffer2 [6*6] = {}; 
      float buffer3 [3*6] = {}; 
      float buffer4 [3*3] = {}; 
      Mat<float> lambda_noe_noe_dash(buffer1, 6, 6); 
      Mat<float> lambda_noe_noe_dashinv(buffer2, 6, 6);
      Mat<float> lambdaprod(buffer3, 3, 6);
      Mat<float> lambda_noe_noe_sum(buffer4, 3, 3);

      // Calculate message lambda by marginalisation
      matSum(f_noe_noe_lambda, belief_nonoutedge_lambda, lambda_noe_noe_dash);  // Sum block of fp lambdamatrix and belief lambda
      matSum(lambda_noe_noe_dash, pmess_nonoutedge_lambda, lambda_noe_noe_dash, true);  // Subtract from above the previous message lambda

      inv6x6(lambda_noe_noe_dash, lambda_noe_noe_dashinv);
      matMul(f_oe_noe_lambda, lambda_noe_noe_dashinv, lambdaprod);

      matMul(lambdaprod, f_noe_oe_lambda, lambda_noe_noe_sum);

      matSum(f_oe_oe_lambda, lambda_noe_noe_sum, mess_outedge_lambda, true);  

    }
    else {
      for (unsigned i = 0; i < mess_outedge_lambda_.size(); ++i) {
        mess_outedge_lambda_[i] = 0.0;
      }
    } 

    return true;
  }
};
