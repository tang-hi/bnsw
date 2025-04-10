#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "dist_alg/l2_distance.hpp"
#include <random>

TEST_CASE("L2 Distance Calculation", "[l2_distance]") {
  const int dim = 8;
  float a[dim] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
  float b[dim] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

  L2Distance<float> l2_distance;
  using Catch::Matchers::WithinRel;
  // Test with identical vectors
  REQUIRE_THAT(l2_distance.distance(a, b, dim), WithinRel(0.0f, 1e-5f));
  // Test with different vectors
  b[7] = 9.0;
  REQUIRE_THAT(l2_distance.distance(a, b, dim), WithinRel(1.0f, 1e-5f));
}

TEST_CASE("L2 Distance Calculation", "unaligned") {
  const int dim = 13;
  float a[dim];
  float b[dim];
  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(-100.0, 100.0);
  float L2 = 0.0;
  for (auto i = 0; i < dim; ++i) {
    a[i] = distribution(generator);
    b[i] = distribution(generator);
    L2 += (a[i] - b[i]) * (a[i] - b[i]);
  }
  L2 = std::sqrt(L2);

  L2Distance<float> l2_distance;
  using Catch::Matchers::WithinRel;
  // Test with identical vectors
  REQUIRE_THAT(l2_distance.distance(a, b, dim), WithinRel(L2, 1e-5f));
}
