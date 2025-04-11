#pragma once

#include "Eigen/Dense"
#include "dist_alg/l2_distance.hpp"
#include "utils.hpp"
#include <cmath>
#include <cstddef>
#include <type_traits>
namespace bnsw {

template <typename T,
          template <typename> typename DistanceAlgorithm = L2Distance>
class AdSampling {
  static_assert(std::is_default_constructible_v<DistanceAlgorithm<T>>,
                "DistanceAlgorithm must be default constructible.");
  static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");

public:
  explicit AdSampling(int dimension)
      : orthogonal_matrix(createOrthogonal(dimension)) {}

  float distance(const T *a, const T *b) {
    return distance_algorithm.distance(a, b, dimension);
  }

  void convert(T *ptr) {
    // (1, dim) * (dim, dim) = (1, dim)
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> vec(ptr, dimension);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> result(
        ptr, dimension); // Reuse the same memory
    result = vec * orthogonal_matrix;
    return;
  }

  bool above_threshold(const T *a, const T *b, float threshold) {
    float estimate = 0.0f;

    for (int i = 0; i < dimension / batch; ++i) {
      const auto *a_ptr = a + i * batch;
      const auto *b_ptr = b + i * batch;
      estimate += distance_algorithm.distance(a_ptr, b_ptr, batch);

      double ratio = (1.0 * i / dimension) *
                     (1.0 + eps0 / std::sqrt(i * batch)) *
                     (1.0 + eps0 / std::sqrt(i * batch));
      if (estimate > threshold * ratio) {
        return true;
      }
    }
    return estimate > threshold;
  }

private:
  int dimension{0};
  Eigen::MatrixXf orthogonal_matrix;
  DistanceAlgorithm<T> distance_algorithm;

  constexpr static const int batch = 8;
  constexpr static const double eps0 = 2.1;
};

} // namespace bnsw