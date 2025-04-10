#include "catch2/catch_test_macros.hpp"
#include "dist_alg/l2_distance.hpp"
#include "utils.hpp"
#include <algorithm>
#include <queue>
#include <random>
#include <utility>
#include <vector>

TEST_CASE("AdSampling Test", "[predict]") {
  const int dim = 128;
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(-100.0, 100.0);
  // prepare data
  std::vector<std::vector<float>> data(1000, std::vector<float>(dim));

  for (int i = 0; i < 1000; ++i) {
    for (int j = 0; j < dim; ++j) {
      data[i][j] = distribution(generator);
    }
  }

  // prepare query
  std::vector<float> query(dim);
  for (int i = 0; i < dim; ++i) {
    query[i] = distribution(generator);
  }

  L2Distance<float> l2_distance;
  std::vector<std::pair<float, int>> ranked_results;
  ranked_results.reserve(data.size());
  for (auto i = 0U; i < data.size(); ++i) {
    float distance = l2_distance.distance(query.data(), data[i].data(), dim);
    ranked_results.emplace_back(distance, i);
  }

  // raw ranking results
  std::sort(ranked_results.begin(), ranked_results.end(),
            [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
              return a.first < b.first;
            });

  // transform the raw vector
  auto orthogonal = createOrthogonal(dim);
  auto transformer = [&](std::vector<float> &vec) -> std::vector<float> {
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> v(vec.data(), dim);
    Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, 1>> r(
        vec.data(), dim); // Reuse the same memory
    r = v * orthogonal;
    return vec;
  };
  std::vector<float> transformed_query = transformer(query);
  std::transform(data.begin(), data.end(), data.begin(), transformer);

  std::priority_queue<std::pair<float, int>, std::vector<std::pair<float, int>>,
                      std::greater<std::pair<float, int>>>
      pq;

  auto threshold = 0.0f;
  for (auto i = 0U; i < data.size(); ++i) {
  }
}