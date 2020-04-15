#include <iostream>


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
