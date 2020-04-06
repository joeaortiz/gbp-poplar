#include <poplar/Vertex.hpp>

using namespace poplar;

class Mat{
  
  private:
    unsigned rows_, cols_;
    int rowstride_;
    float * p_;

  public:
    // Constructor  which is automatically called when the object is created. 
    // Must have same name as the class type.
    Mat(float * p, unsigned rows, unsigned cols, int rowstride = 0) 
    {
      p_ = p;
      rows_ = rows;
      cols_ = cols;
      rowstride_ = rowstride; // How many entries in memory each row is separated by
    }

    // Get functions
    unsigned getnCols() const
    {
      return cols_;
    }

    unsigned getnRows() const
    {
      return rows_;
    }

    //Overload () operator to retrieve the element of the matrix at address p in row r and column c
    float& operator()(unsigned r, unsigned c)
    {
      return p_[r * (cols_ + rowstride_) + c];
    }

    const float& operator()(unsigned r, unsigned c) const
    {
      return  p_[r * (cols_ + rowstride_) + c];
    }

};


void matMul(const Mat& Mat1, const Mat& Mat2, Mat& product)
{
  // // you could use a constructor of Mat that alreay allocates the memory for you
  // float* array = new float[this->rows * otherMat.getnCols()]; // allocate mem for result
  // Mat res(array, this->rows, otherMat.getnCols()); // create a matrix view
  
  if ( Mat1.getnCols() == Mat2.getnRows() ) {  // Matrix dimensions must match up.

    for (unsigned i = 0; i < Mat1.getnRows(); ++i) {
      for (unsigned j = 0; j < Mat2.getnCols(); ++j) {
        product(i,j) = 0;
        for (unsigned k = 0; k < Mat1.getnCols(); ++k) {
          product(i, j) += Mat1(i, k) * Mat2(k,j);
        }
      }
    }
  }
  else {  // Matrix multiplication not possible
    for (unsigned i = 0; i < Mat1.getnRows(); ++i) {
      for (unsigned j = 0; j < Mat2.getnCols(); ++j) {
        product(i, j) = -1;  // Fill matrix with -1s to indicate matrix multiplication was not valid
      }
    }
  }

}

void matSum(const Mat& Mat1, const Mat& Mat2, Mat& sum, bool subtract = false)
{
  // // you could use a constructor of Mat that alreay allocates the memory for you
  // float* array = new float[this->rows * otherMat.getnCols()]; // allocate mem for result
  // Mat res(array, this->rows, otherMat.getnCols()); // create a matrix view
  
  // Matrix dimensions must be equal.
  if (( Mat1.getnRows() == Mat2.getnRows() ) && ( Mat1.getnCols() == Mat2.getnCols() )) {

    for (unsigned i = 0; i < Mat1.getnRows(); ++i) {
      for (unsigned j = 0; j < Mat1.getnCols(); ++j) {
        if (subtract == false){
          sum(i,j) = Mat1(i,j) + Mat2(i,j);
        }
        else {
          sum(i,j) = Mat1(i,j) - Mat2(i,j);
        }
      }
    }
  }  
}


void mat2by2Inv(const Mat& matrix, Mat& inverse)
{

  // Matrix must be square.
  if ( matrix.getnRows() == matrix.getnCols() ) {

    float det = matrix(0,0) * matrix(1,1) - matrix(0,1) * matrix(1,0);

    inverse(0,0) = matrix(1,1) / det;
    inverse(1,0) = - matrix(1,0) / det;
    inverse(0,1) = - matrix(0,1) / det;
    inverse(1,1) = matrix(0,0) / det;
  }   
}



// A vertex type to update eta of the belief of a variable node. 
class UpdateBeliefEtaVertex : public Vertex {
public:
  Input<unsigned> var_dofs;
  Input<Vector<float>> prior_eta;
  Input<Vector<float>> incoming_messages_eta;

  Output<Vector<float>> belief_eta;

  bool compute() {
    // Set belief values back to zeros in memory
    for (unsigned i = 0; i < belief_eta.size(); ++i) {
      belief_eta[i] = 0.0;
    }

    // Create matrix objects for belief and prior eta 
    Mat mat_belief_eta(&belief_eta[0], var_dofs, 1);
    Mat mat_prior_eta(&prior_eta[0], var_dofs, 1);

    // Add priors to belief
    matSum(mat_belief_eta, mat_prior_eta, mat_belief_eta);

    // Add incoming messages from all edges. 
    unsigned nincoming_mess = incoming_messages_eta.size() / var_dofs;
    for (unsigned i = 0; i < nincoming_mess; ++i) {
      Mat incoming_eta(&incoming_messages_eta[i * 2], var_dofs, 1);
      matSum(mat_belief_eta, incoming_eta, mat_belief_eta);
    }
    return true;
  }
};

// A vertex type to update lambda of the belief of a variable node. 
class UpdateBeliefLambdaVertex : public Vertex {
public:
  Input<unsigned> var_dofs;
  Input<Vector<float>> prior_lambda;
  Input<Vector<float>> incoming_messages_lambda;

  Output<Vector<float>> belief_lambda;

  bool compute() {
    // Set belief values back to zeros in memory
    for (unsigned i = 0; i < belief_lambda.size(); ++i) {
      belief_lambda[i] = 0.0;
    }

    // Create matrix objects for belief and prior lambda
    Mat mat_belief_lambda(&belief_lambda[0], var_dofs, var_dofs);
    Mat mat_prior_lambda(&prior_lambda[0], var_dofs, var_dofs);

    // Add priors to belief
    matSum(mat_belief_lambda, mat_prior_lambda, mat_belief_lambda);

    // Add incoming messages from all edges. 
    unsigned nincoming_mess = incoming_messages_lambda.size() / (var_dofs * var_dofs);
    for (unsigned i = 0; i < nincoming_mess; ++i) {
      Mat incoming_lambda(&incoming_messages_lambda[i * 4], var_dofs, var_dofs);
      matSum(mat_belief_lambda, incoming_lambda, mat_belief_lambda);
    }
    return true;
  }
};


// A vertex type to compute the eta message going to the node variables first in the factor potential
class ComputeMessageEtaVertex : public Vertex {
public:
  // The inputs read a vector of values (i.e. an ordered, contiguous set of values in memory) from the graph.
  // Must input beliefs separately as the beliefs vector is ordered in order of variableID
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
  Output<Vector<float>> mess_outedge_eta_;  

  bool compute() {

    // Create input matrix objects
    Mat f_outedge_eta(&f_outedge_eta_[0], outedge_dofs, 1); 
    Mat f_nonoutedge_eta(&f_nonoutedge_eta_[0], nonoutedge_dofs, 1);
    Mat f_noe_noe_lambda(&f_noe_noe_lambda_[0], nonoutedge_dofs, nonoutedge_dofs);
    Mat f_oe_noe_lambda(&f_oe_noe_lambda_[0], outedge_dofs, nonoutedge_dofs);

    Mat belief_nonoutedge_eta(&belief_nonoutedge_eta_[0], nonoutedge_dofs, 1);
    Mat belief_nonoutedge_lambda(&belief_nonoutedge_lambda_[0], nonoutedge_dofs, nonoutedge_dofs);  

    Mat pmess_nonoutedge_eta(&pmess_nonoutedge_eta_[0], nonoutedge_dofs, 1);
    Mat pmess_nonoutedge_lambda(&pmess_nonoutedge_lambda_[0], nonoutedge_dofs, nonoutedge_dofs);
    Mat pmess_outedge_eta(&pmess_outedge_eta_[0], outedge_dofs, 1);

    // Create output matrix object
    Mat mess_outedge_eta(&mess_outedge_eta_[0], outedge_dofs, 1);

    // Create buffers to hold intermediate data in calculation
    float buffer1 [4]; 
    float buffer2 [4]; 
    float buffer3 [4]; 
    float buffer4 [2]; 
    float buffer5 [2]; 
    Mat lambda_noe_noe_dash(buffer1, 2, 2);  // Test what happens when I put the input float here instead of 2
    Mat lambda_noe_noe_dashinv(buffer2, 2, 2);
    Mat lambdaprod(buffer3, 2, 2);
    Mat eta_noe_dash(buffer4, 2, 1);
    Mat eta_noe_sum(buffer5, 2, 1);

    // Calculate message eta by marginalisation
    matSum(f_noe_noe_lambda, belief_nonoutedge_lambda, lambda_noe_noe_dash);  // Sum block of fp lambdamatrix and belief lambda
    matSum(lambda_noe_noe_dash, pmess_nonoutedge_lambda, lambda_noe_noe_dash, true);  // Subtract from above the previous message lambda

    mat2by2Inv(lambda_noe_noe_dash, lambda_noe_noe_dashinv);
    matMul(f_oe_noe_lambda, lambda_noe_noe_dashinv, lambdaprod);

    matSum(f_nonoutedge_eta, belief_nonoutedge_eta, eta_noe_dash);  // Sum block of fp lambdamatrix and belief lambda
    matSum(eta_noe_dash, pmess_nonoutedge_eta, eta_noe_dash, true);  // Subtract from above the previous message lambda

    matMul(lambdaprod, eta_noe_dash, eta_noe_sum);

    matSum(f_outedge_eta, eta_noe_sum, mess_outedge_eta, true);

    return true;
  }
};


// A vertex type to compute the messages from a factor node. 
class ComputeMessageLambdaVertex : public Vertex {
public:
  // The inputs read a vector of values (i.e. an ordered, contiguous set of values in memory) from the graph.
  // Must input beliefs separately as the beliefs vector is ordered in order of variableID
  Input<unsigned> outedge_dofs; 
  Input<unsigned> nonoutedge_dofs; 

  Input<Vector<float>> f_oe_oe_lambda_;
  Input<Vector<float>> f_noe_noe_lambda_;
  Input<Vector<float>> f_oe_noe_lambda_;
  Input<Vector<float>> f_noe_oe_lambda_;

  Input<Vector<float>> belief_nonoutedge_lambda_;

  // Vectors containing previous outward messages from the factor node
  // Must input them as separate vectors as the out_messages vector is ordered by variable node it is sent to rather than by factor node.
  // For computing message  
  Input<Vector<float>> pmess_nonoutedge_lambda_;  
  // For message damping
  Input<Vector<float>> pmess_outedge_lambda_;


  // The ouput is to a vector in the graph which is the new outgoing messages from the factor node. 
  Output<Vector<float>> mess_outedge_lambda_;  

  bool compute() {

    // Create input matrix objects
    Mat f_oe_oe_lambda(&f_oe_oe_lambda_[0], outedge_dofs, outedge_dofs); 
    Mat f_noe_noe_lambda(&f_noe_noe_lambda_[0], nonoutedge_dofs, nonoutedge_dofs);
    Mat f_oe_noe_lambda(&f_oe_noe_lambda_[0], outedge_dofs, nonoutedge_dofs);
    Mat f_noe_oe_lambda(&f_noe_oe_lambda_[0], nonoutedge_dofs, outedge_dofs);

    Mat belief_nonoutedge_lambda(&belief_nonoutedge_lambda_[0], nonoutedge_dofs, nonoutedge_dofs);  

    Mat pmess_nonoutedge_lambda(&pmess_nonoutedge_lambda_[0], nonoutedge_dofs, nonoutedge_dofs);
    Mat pmess_outedge_lambda(&pmess_outedge_lambda_[0], outedge_dofs, outedge_dofs);

    // Create output matrix object
    Mat mess_outedge_lambda(&mess_outedge_lambda_[0], outedge_dofs, outedge_dofs);

    // Create buffers to hold intermediate data in calculation
    float buffer1 [4]; 
    float buffer2 [4]; 
    float buffer3 [4]; 
    float buffer4 [4]; 
    Mat lambda_noe_noe_dash(buffer1, 2, 2);  // Test what happens when I put the input float here instead of 2
    Mat lambda_noe_noe_dashinv(buffer2, 2, 2);
    Mat lambdaprod(buffer3, 2, 2);
    Mat lambda_noe_noe_sum(buffer4, 2, 2);

    // Calculate message lambda by marginalisation
    matSum(f_noe_noe_lambda, belief_nonoutedge_lambda, lambda_noe_noe_dash);  // Sum block of fp lambdamatrix and belief lambda
    matSum(lambda_noe_noe_dash, pmess_nonoutedge_lambda, lambda_noe_noe_dash, true);  // Subtract from above the previous message lambda

    mat2by2Inv(lambda_noe_noe_dash, lambda_noe_noe_dashinv);
    matMul(f_oe_noe_lambda, lambda_noe_noe_dashinv, lambdaprod);

    matMul(lambdaprod, f_noe_oe_lambda, lambda_noe_noe_sum);

    matSum(f_oe_oe_lambda, lambda_noe_noe_sum, mess_outedge_lambda, true);  

    return true;
  }
};
