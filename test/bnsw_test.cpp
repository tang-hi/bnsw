#include "bnsw.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <random>
#include <vector>

// Test distance function for 1D points
template <typename T> struct TestDistance {
  float distance(const T *a, const T *b, int) const {
    return std::abs(a[0] - b[0]);
  }
};

namespace bnsw {

// Accessor to test private functions
struct BnswTestAccessor {
  template <typename T, template <typename> class DA,
            template <typename, template <typename> class> class SA>
  static auto searchLayer(const bnsw<T, DA, SA> &index, const T *query,
                          std::uint32_t entry_point, int level,
                          std::size_t ef) {
    return index.searchLayer(query, entry_point, level, ef);
  }

  template <typename T, template <typename> class DA,
            template <typename, template <typename> class> class SA>
  static void addNode(bnsw<T, DA, SA> &index,
                      typename bnsw<T, DA, SA>::InternalNode node,
                      typename bnsw<T, DA, SA>::label_t label) {
    index.nodes_.push_back(node);
    index.id_to_label_[index.nodes_.size() - 1] = label;
    index.label_to_id_[label] = index.nodes_.size() - 1;
    index.id_to_data_[index.nodes_.size() - 1] = node.point_data;
  }

  template <typename T, template <typename> class DA,
            template <typename, template <typename> class> class SA>
  static void pruneNeighbors(
      bnsw<T, DA, SA> &index,
      std::priority_queue<typename bnsw<T, DA, SA>::Neighbor,
                          std::vector<typename bnsw<T, DA, SA>::Neighbor>,
                          std::greater<typename bnsw<T, DA, SA>::Neighbor>>
          &candidates_min_heap,
      const int limits) {
    index.pruneNeighbors(candidates_min_heap, limits);
  }

  template <typename T, template <typename> class DA,
            template <typename, template <typename> class> class SA>
  static id_t selectAndConnectNeighbors(
      bnsw<T, DA, SA> &index, id_t current_id,
      std::priority_queue<typename bnsw<T, DA, SA>::Neighbor,
                          std::vector<typename bnsw<T, DA, SA>::Neighbor>,
                          std::greater<typename bnsw<T, DA, SA>::Neighbor>>
          &candidates_min_heap,
      size_t M, const int level) {
    return index.selectAndConnectNeighbors(current_id, candidates_min_heap, M,
                                           level);
  }
};

TEST_CASE("Bnsw Exception Handling", "[duplicate label]") {
  std::vector<int> vec(128, 0);
  bnsw<float, TestDistance> bnsw_instance(128, 16, 16, 16);

  REQUIRE(bnsw_instance.addPoint(vec.data(), 0));
  REQUIRE_FALSE(bnsw_instance.addPoint(vec.data(), 0));
}

TEST_CASE("Bnsw searchLayer functionality",
          "[searchLayer ef=1 returns nearest neighbors]") {
  const int dim = 1;
  const size_t M = 4;
  const size_t ef = 1;
  const int seed = 42;

  // Create a small graph with 5 points in 1D for simplicity
  bnsw<float, TestDistance> index(dim, M, ef, ef, seed);
  std::vector<float> points = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  for (size_t i = 0; i < points.size(); ++i) {
    bnsw<float, TestDistance>::InternalNode node(0, &points[i], M, M);
    // process node's neighbors
    for (size_t j = 0; j < points.size(); ++j) {
      if (i != j) {
        node.connections[0].push_back(j);
      }
    }
    BnswTestAccessor::addNode(index, node, i);
  }

  using Catch::Matchers::WithinRel;
  auto result = BnswTestAccessor::searchLayer(index, &points[0],
                                              points.size() - 1, 0, ef);
  REQUIRE(result.size() == 1);
  REQUIRE(result.top().id == 0);

  float query = 2.6f;
  result =
      BnswTestAccessor::searchLayer(index, &query, points.size() - 1, 0, ef);
  REQUIRE(result.size() == 1);
  REQUIRE(result.top().id == 3);
  REQUIRE_THAT(result.top().distance, WithinRel(0.4f, 1e-5f));
}

TEST_CASE("Bnsw searchLayer functionality", "[random test]") {
  const int dim = 1;
  const size_t M = 16;
  const size_t ef = 10;
  const int seed = 42;
  const int element_count = 3000;
  const float tolerance = 1e-5f;

  using namespace Catch::Matchers;
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<float> distribution(0.0, 100.0);

  bnsw<float, TestDistance> index(dim, M, ef, ef, seed);
  std::vector<float> points;
  points.reserve(element_count);
  for (int i = 0; i < element_count; i++) {
    float ra = distribution(generator);
    points.push_back(ra);
  }

  // mock the bottom level of the graph, where each node has 2 * M neighbors
  // and each neighbor is the nearest neighbor of the point
  for (int i = 0; i < element_count; i++) {
    bnsw<float, TestDistance>::InternalNode node(0, &points[i], M, 2 * M);
    std::priority_queue<std::pair<float, uint32_t>,
                        std::vector<std::pair<float, uint32_t>>,
                        std::greater<std::pair<float, uint32_t>>>
        top_k_neighbors;
    for (auto idx = 0; idx < element_count; idx++) {
      if (idx != i) {
        top_k_neighbors.emplace(std::abs(points[i] - points[idx]), idx);
      }
    }
    for (auto count = 0U; count < 2 * M; count++) {
      auto [distance, idx] = top_k_neighbors.top();
      top_k_neighbors.pop();
      node.connections[0].push_back(idx);
    }
    BnswTestAccessor::addNode(index, node, i);
  }

  auto search_num = 30U;
  for (const auto &point : points) {
    auto entry =
        std::uniform_int_distribution<>(0, points.size() - 1)(generator);
    auto result =
        BnswTestAccessor::searchLayer(index, &point, entry, 0, search_num);

    int idx = 0;
    std::vector<std::pair<float, uint32_t>> ground_truth;
    ground_truth.resize(points.size());
    std::transform(points.begin(), points.end(), ground_truth.begin(),
                   [&point, &idx](float p) {
                     return std::make_pair(std::abs(point - p), idx++);
                   });
    std::sort(ground_truth.begin(), ground_truth.end());
    REQUIRE(result.size() == search_num);
    while (!result.empty()) {
      auto [id, distance] = result.top();
      auto [gt_distance, gt_idx] = *ground_truth.begin();
      ground_truth.erase(ground_truth.begin());
      result.pop();

      // Check if the distance calculated is correct
      REQUIRE_THAT(distance, WithinRel(gt_distance, tolerance));
    }
  }
}

TEST_CASE("Bnsw prune Neighbors functionality", "[test heuristic]") {}

TEST_CASE("Bnsw selectAndConnectNeighbors functionality",
          "[selectAndConnectNeighbors]") {}

TEST_CASE("Bnsw search functionality", "[search-recall]") {}

}; // namespace bnsw
