#include "bnsw.hpp"
#include "catch2/catch_test_macros.hpp"
#include "catch2/matchers/catch_matchers_floating_point.hpp"
#include "dist_alg/l2_distance.hpp"
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

  template <typename T, template <typename> class DA,
            template <typename, template <typename> class> class SA>
  static const auto &getNodeConnections(const bnsw<T, DA, SA> &index,
                                        id_t node_id, int level) {
    return index.nodes_[node_id].connections[level];
  }

  template <typename T, template <typename> class DA,
            template <typename, template <typename> class> class SA>
  static auto &getMutableNodeConnections(bnsw<T, DA, SA> &index, id_t node_id,
                                         int level) {
    return index.nodes_[node_id].connections[level];
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

TEST_CASE("Bnsw prune Neighbors functionality", "[less than limits]") {
  const int dim = 1;
  const size_t M = 4;
  const size_t ef = 1;
  const int seed = 42;
  using Neighbor = bnsw<float, TestDistance>::Neighbor;
  bnsw<float, TestDistance> index(dim, M, ef, ef, seed);
  std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>>
      candidates_min_heap;
  candidates_min_heap.emplace(0, 0);
  candidates_min_heap.emplace(1, 1);

  BnswTestAccessor::pruneNeighbors(index, candidates_min_heap, 3);

  // less than limits, should not prune
  REQUIRE(candidates_min_heap.size() == 2);
}

TEST_CASE("bnsw prune Neighbors functionality", "[heuristic pruning]") {
  const int dim = 1;
  const size_t M = 4;
  const size_t ef = 1;
  const int seed = 42;
  using Neighbor = bnsw<float, TestDistance>::Neighbor;
  bnsw<float, TestDistance> index(dim, M, ef, ef, seed);
  std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>>
      candidates_min_heap;

  std::vector<float> vec(4, 0);
  // add some node
  for (int i = 0; i < 4; ++i) {
    vec[i] = i;
    bnsw<float, TestDistance>::InternalNode node(0, &vec[i], M, 2 * M);
    BnswTestAccessor::addNode(index, node, i);
  }

  candidates_min_heap.emplace(0, 0.0);
  candidates_min_heap.emplace(1, 0.8);
  candidates_min_heap.emplace(2, 2.1);
  candidates_min_heap.emplace(3, 0.9);

  BnswTestAccessor::pruneNeighbors(index, candidates_min_heap, 3);

  // should prune to size of limits
  REQUIRE(candidates_min_heap.size() == 3);

  // 2 should be pruned
  while (!candidates_min_heap.empty()) {
    auto candidate = candidates_min_heap.top();
    REQUIRE(candidate.id != 2);
    candidates_min_heap.pop();
  }
}

TEST_CASE("Bnsw selectAndConnectNeighbors functionality",
          "[selectAndConnectNeighbors]") {
  const int dim = 1;
  const size_t M = 2;  // Max connections per node
  const size_t ef = 1; // Not directly used here, but needed for constructor
  const int seed = 42;
  using Neighbor = bnsw<float, TestDistance>::Neighbor;
  bnsw<float, TestDistance> index(dim, M, ef, ef, seed);

  // Add existing nodes to the index
  std::vector<float> points = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
  for (size_t i = 0; i < points.size(); ++i) {
    // Initialize nodes with M max connections at level 0
    bnsw<float, TestDistance>::InternalNode node(0, &points[i], M, M);
    BnswTestAccessor::addNode(index, node, i);
  }

  // Pre-populate node 3 (point 3.0f) with M neighbors (0 and 1) to test pruning
  auto &node3_initial_connections =
      BnswTestAccessor::getMutableNodeConnections(index, 3, 0);
  node3_initial_connections.push_back(0); // Distance |3.0 - 0.0| = 3.0
  node3_initial_connections.push_back(1); // Distance |3.0 - 1.0| = 2.0
  REQUIRE(node3_initial_connections.size() == M);

  // Simulate adding a new node (id 5) with point 2.5f
  float new_point_data = 2.5f;
  id_t current_id = points.size(); // New node ID will be 5
  bnsw<float, TestDistance>::InternalNode new_node(0, &new_point_data, M, M);
  BnswTestAccessor::addNode(index, new_node, current_id);

  // Create candidate neighbors for the new node (id 5)
  // Neighbors are (id, distance) from the new point 2.5f
  std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>>
      candidates_min_heap;
  candidates_min_heap.emplace(2, 0.5f); // Node 2, distance |2.0 - 2.5| = 0.5
  candidates_min_heap.emplace(3, 0.5f); // Node 3, distance |3.0 - 2.5| = 0.5
  candidates_min_heap.emplace(1, 1.5f); // Node 1, distance |1.0 - 2.5| = 1.5
  candidates_min_heap.emplace(4, 1.5f); // Node 4, distance |4.0 - 2.5| = 1.5
  candidates_min_heap.emplace(0, 2.5f); // Node 0, distance |0.0 - 2.5| = 2.5

  // Call the function under test for level 0
  BnswTestAccessor::selectAndConnectNeighbors(index, current_id,
                                              candidates_min_heap, M, 0);

  // Verify connections for the new node (id 5)
  const auto &new_node_connections =
      BnswTestAccessor::getNodeConnections(index, current_id, 0);
  REQUIRE(new_node_connections.size() == M); // Should have M=2 neighbors

  // Check if the closest neighbors (2 and 3) were selected
  std::vector<id_t> neighbor_ids;
  std::transform(new_node_connections.begin(), new_node_connections.end(),
                 std::back_inserter(neighbor_ids),
                 [](const auto &neighbor_id) { return neighbor_id; });
  std::sort(neighbor_ids.begin(), neighbor_ids.end());
  REQUIRE(neighbor_ids[0] == 2);
  REQUIRE(neighbor_ids[1] == 3);

  // Verify connections for the selected neighbors (2 and 3) - they should now
  // connect back to 5
  const auto &neighbor2_connections =
      BnswTestAccessor::getNodeConnections(index, 2, 0);
  REQUIRE(std::find(neighbor2_connections.begin(), neighbor2_connections.end(),
                    current_id) != neighbor2_connections.end());

  // Verify connections for node 3 - it should now connect back to 5
  // and its connections should have been pruned.
  const auto &neighbor3_connections =
      BnswTestAccessor::getNodeConnections(index, 3, 0);
  REQUIRE(neighbor3_connections.size() == 1); // Still M connections
  REQUIRE(std::find(neighbor3_connections.begin(), neighbor3_connections.end(),
                    current_id) !=
          neighbor3_connections.end()); // Should contain the new node 5

  // Verify connections for nodes that should NOT have been selected (0, 1, 4)
  const auto &neighbor0_connections =
      BnswTestAccessor::getNodeConnections(index, 0, 0);
  REQUIRE(std::find(neighbor0_connections.begin(), neighbor0_connections.end(),
                    current_id) == neighbor0_connections.end());

  const auto &neighbor1_connections =
      BnswTestAccessor::getNodeConnections(index, 1, 0);
  REQUIRE(std::find(neighbor1_connections.begin(), neighbor1_connections.end(),
                    current_id) == neighbor1_connections.end());

  const auto &neighbor4_connections =
      BnswTestAccessor::getNodeConnections(index, 4, 0);
  REQUIRE(std::find(neighbor4_connections.begin(), neighbor4_connections.end(),
                    current_id) == neighbor4_connections.end());
}

TEST_CASE("Bnsw search functionality", "[search-recall]") {
  const int dim = 1;
  const size_t M = 16;
  const size_t ef = 40;
  const int seed = 42;
  const int element_count = 30000;
  std::vector<std::pair<float, int>> points;
  points.reserve(element_count);
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<float> distribution(0.0, 1000.0);
  for (int i = 0; i < element_count; i++) {
    float ra = distribution(generator);
    points.push_back({ra, i});
  }

  bnsw<float, L2Distance> index(dim, M, ef, ef, seed);
  for (const auto &point : points) {
    index.addPoint(&point.first, point.second);
  }

  const int search_query_count = 100;
  const uint32_t k = 10;

  for (int i = 0; i < search_query_count; i++) {
    auto query_index =
        std::uniform_int_distribution<>(0, points.size() - 1)(generator);
    auto query = points[query_index].first;
    auto result = index.search(&query, k);
    REQUIRE(result.size() == k);

    // Check if the results are unique
    std::unordered_set<int> unique_results(result.begin(), result.end());
    REQUIRE(unique_results.size() == k);

    // get the ground truth
    auto points_copy = points;
    std::sort(points_copy.begin(), points_copy.end(),
              [&query](const auto &a, const auto &b) {
                return std::abs(a.first - query) < std::abs(b.first - query);
              });

    std::vector<int> ground_truth;
    ground_truth.reserve(k);
    for (auto j = 0u; j < k; ++j) {
      ground_truth.push_back(points_copy[j].second);
    }

    float recalled = 0;
    // Check if the results are in the ground truth
    for (const auto &result_id : result) {
      if (std::find(ground_truth.begin(), ground_truth.end(), result_id) !=
          ground_truth.end()) {
        recalled++;
      }
    }

    float recall_rate = recalled / k;
    REQUIRE(recall_rate >= 0.95f);
  }
}

}; // namespace bnsw
