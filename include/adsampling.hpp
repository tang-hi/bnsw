#pragma once

#include "Eigen/Dense"
#include "dist_alg/l2_distance.hpp"
#include <cmath>
#include <type_traits>
#include <vector>

namespace bnsw {

template <typename T,
          template <typename> typename DistanceAlgorithm = L2Distance>
class AdSampling {
  static_assert(std::is_default_constructible_v<DistanceAlgorithm<T>>,
                "DistanceAlgorithm must be default constructible.");
  static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");

public:
  static constexpr bool need_convert = true;
  explicit AdSampling(int dimension) : dimension(dimension) {
    // Precompute r threshold factors
    int num_steps = dimension / batch;
    if (num_steps > 0) {
      r_threshold_factors.resize(num_steps);
      for (int i = 0; i < num_steps; ++i) {
        double current_dimension = static_cast<double>((i + 1) * batch);
        double factor = 1.0 + eps0 / std::sqrt(current_dimension);
        r_threshold_factors[i] =
            (current_dimension / this->dimension) * factor * factor;
      }
    }
  }

  float distance(const T *a, const T *b) {
    return distance_algorithm.distance(a, b, dimension);
  }

  void set_orthogonal_matrix(const Eigen::MatrixXf *matrix) {
    orthogonal_matrix = matrix;
  }

  T *convert(const T *ptr) const {
    // (1, dim) * (dim, dim) = (1, dim)
    T *result = new T[dimension];
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> vec(ptr, dimension);
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>> result_map(
        result, dimension); // Use the new memory
    // Dereference the pointer before multiplication
    result_map = (*orthogonal_matrix) * vec;
    return result; // Return the new pointer
  }

  bool above_threshold(const T *a, const T *b, float threshold,
                       float &estimate) const {
    if (dimension < batch) {
      estimate = distance_algorithm.distance(a, b, dimension);
      return estimate > threshold;
    }
    estimate = 0.0f;
    int i = 0;
    for (; i < dimension / batch; ++i) {
      const auto *a_ptr = a + i * batch;
      const auto *b_ptr = b + i * batch;
      estimate += distance_algorithm.distance(a_ptr, b_ptr, batch);
      auto current_dimension = (i + 1) * batch;
      double r = r_threshold_factors[i];
      if (current_dimension < dimension && estimate > threshold * r) {
        early_stop_count += 1;
        estimate = estimate * dimension / current_dimension;
        return true;
      }

      if (i == dimension / batch - 1) {
        // Handle the last batch
        int remaining_dimension = dimension - (i + 1) * batch;
        estimate += distance_algorithm.distance(
            a + (i + 1) * batch, b + (i + 1) * batch, remaining_dimension);
      }
    }

    return estimate > threshold;
  }

public:
  mutable int early_stop_count{0};

private:
  int dimension{0};
  const Eigen::MatrixXf *orthogonal_matrix{nullptr};
  DistanceAlgorithm<T> distance_algorithm;
  std::vector<double> r_threshold_factors; // Store precomputed r values
  constexpr static const int batch = 128;
  constexpr static const double eps0 = 2.1;
};

} // namespace bnsw