#pragma once

#include "Eigen/Dense"
#include "dist_alg/l2_distance.hpp"
#include <cmath>

namespace bnsw {
template <typename T,
          template <typename> typename DistanceAlgorithm = L2Distance>
class NonSampling {
public:
  explicit NonSampling(int dimension) : dimension(dimension) {}

  static constexpr bool need_convert = true;

  float distance(const T *a, const T *b) {
    return distance_algorithm.distance(a, b, dimension);
  }

  T *convert(const T *ptr) const {
    T *result = new T[dimension];
    std::copy(ptr, ptr + dimension, result);
    return result; // Return the new pointer
  }

  bool above_threshold(const T *a, const T *b, float threshold,
                       float &estimate) const {
    estimate = distance_algorithm.distance(a, b, dimension);
    return estimate >= threshold;
  }

  void set_orthogonal_matrix(const Eigen::MatrixXf *) {
    return;
  }


  int get_early_stop_count() const { return 0; }

private:
  int dimension{0};
  DistanceAlgorithm<T> distance_algorithm;
};
} // namespace bnsw