#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "dist_alg/l2_distance.hpp"

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
