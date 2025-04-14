#pragma once

#include "adsampling.hpp"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bnsw {

template <typename T, template <typename U> class DistanceAlgorithm,
          template <typename V,
                    template <typename> typename> class SamplingAlgorithm =
              AdSampling>
class bnsw {
  using id_t = std::uint32_t;
  using label_t = std::uint32_t;

  static constexpr id_t INVALID_ID = std::numeric_limits<id_t>::max();

  struct InternalNode {
    int level;
    const T *point_data;
    std::vector<std::vector<id_t>> connections;

    InternalNode(int lvl, const T *data_ptr, std::size_t M_max_per_level,
                 std::size_t M_max0)
        : level(lvl), point_data(data_ptr) {
      connections.resize(level + 1);
      for (int i = 1; i <= level; ++i) {
        connections[i].reserve(M_max_per_level);
      }
      if (level >= 0) {
        connections[0].reserve(M_max0);
      }
    }
  };

  struct Neighbor {
    id_t id;
    float distance;

    Neighbor(id_t i, float d) : id(i), distance(d) {}

    bool operator>(const Neighbor &other) const {
      return distance > other.distance;
    }

    bool operator<(const Neighbor &other) const {
      return distance < other.distance;
    }
  };

public:
  explicit bnsw(int dimension, std::size_t M = 16,
                std::size_t ef_construction = 200, std::size_t ef_search = 100,
                int random_seed = 100)
      : dimension_(dimension), M_(M), M_max_(M), M_max0_(M * 2),
        ef_construction_(ef_construction), ef_search_(ef_search),
        level_generator_(random_seed), entry_point_(INVALID_ID), max_level_(-1),
        dist_algo_(dimension), sampler_(dimension),
        mult_(1.0 / std::log(1.0 * M)) {
    if (M == 0)
      throw std::invalid_argument("M cannot be 0");
    if (ef_construction == 0)
      throw std::invalid_argument("ef_construction cannot be 0");
  }

  ~bnsw() = default;

  void addPoint(const void *point, label_t label) {
    const T *point_typed = static_cast<const T *>(point);
    id_t current_id;
    int current_level;

    if (label_to_id_.count(label)) {
      return;
    }

    current_id = element_count_++;
    label_to_id_[label] = current_id;
    id_to_label_[current_id] = label;
    id_to_data_[current_id] = point_typed;
    current_level = getRandomLevel(mult_);
    nodes_.emplace_back(current_level, point_typed, M_max_, M_max0_);

    id_t current_entry_point = entry_point_;
    int current_max_level = max_level_;

    if (current_entry_point == INVALID_ID) {
      entry_point_ = current_id;
      max_level_ = current_level;
      return;
    }

    id_t nearest_node = current_entry_point;
    for (int level = current_max_level; level > current_level; --level) {
      nearest_node = searchLayer(point_typed, nearest_node, level, 1).top().id;
    }

    for (int level = std::min(current_level, current_max_level); level >= 0;
         --level) {
      auto neighbors_min_heap =
          searchLayer(point_typed, nearest_node, level, ef_construction_);
      size_t M_level = (level == 0) ? M_max0_ : M_max_;
      selectAndConnectNeighbors(current_id, neighbors_min_heap, M_level, level);

      std::vector<Neighbor> neighbors_vec;
      while (!neighbors_min_heap.empty()) {
        neighbors_vec.push_back(neighbors_min_heap.top());
        neighbors_min_heap.pop();
      }

      for (const auto &neighbor : neighbors_vec) {
        if (nodes_[neighbor.id].connections.size() <= level)
          continue;

        auto &neighbor_connections = nodes_[neighbor.id].connections[level];
        size_t neighbor_M_level = (level == 0) ? M_max0_ : M_max_;

        if (neighbor_connections.size() > neighbor_M_level) {
          std::priority_queue<Neighbor> candidates_max_heap;
          candidates_max_heap.emplace(current_id,
                                      getDistance(neighbor.id, current_id));

          for (id_t conn_id : neighbor_connections) {
            if (conn_id == current_id)
              continue;
            candidates_max_heap.emplace(conn_id,
                                        getDistance(neighbor.id, conn_id));
          }

          neighbor_connections.clear();
          neighbor_connections.reserve(neighbor_M_level);
          while (neighbor_connections.size() < neighbor_M_level &&
                 !candidates_max_heap.empty()) {
            neighbor_connections.push_back(candidates_max_heap.top().id);
            candidates_max_heap.pop();
          }
        }
      }
    }

    if (current_level > current_max_level) {
      max_level_ = current_level;
      entry_point_ = current_id;
    }
  }

  auto search(const void *query, int k) const -> std::vector<label_t> {
    const T *query_typed = static_cast<const T *>(query);
    id_t current_entry_point = entry_point_;
    int current_max_level = max_level_;

    if (current_entry_point == INVALID_ID || nodes_.empty()) {
      return {};
    }

    id_t nearest_node = current_entry_point;
    for (int level = current_max_level; level > 0; --level) {
      nearest_node = searchLayer(query_typed, nearest_node, level, 1).top().id;
    }

    auto top_candidates_min_heap =
        searchLayer(query_typed, nearest_node, 0, ef_search_);
    std::vector<label_t> results;
    results.reserve(k);

    while (!top_candidates_min_heap.empty() && results.size() < k) {
      id_t candidate_id = top_candidates_min_heap.top().id;
      top_candidates_min_heap.pop();

      if (id_to_label_.count(candidate_id)) {
        results.push_back(id_to_label_.at(candidate_id));
      }
    }

    return results;
  }

private:
  float getDistance(id_t id1, id_t id2) const {
    const T *data1 = id_to_data_.at(id1);
    const T *data2 = id_to_data_.at(id2);
    if (!data1 || !data2) {
      throw std::runtime_error("Invalid data pointer encountered");
    }
    return dist_algo_(data1, data2);
  }

  float getDistance(const T *query, id_t id) const {
    const T *data = id_to_data_.at(id);
    if (!data || !query) {
      throw std::runtime_error("Invalid data pointer encountered");
    }
    return dist_algo_(query, data);
  }

  void selectAndConnectNeighbors(
      id_t current_id,
      std::priority_queue<Neighbor, std::vector<Neighbor>,
                          std::greater<Neighbor>> &candidates_min_heap,
      size_t M, int level) {
    auto &current_connections = nodes_[current_id].connections[level];
    current_connections.reserve(M);

    while (!candidates_min_heap.empty() && current_connections.size() < M) {
      id_t neighbor_id = candidates_min_heap.top().id;
      candidates_min_heap.pop();

      current_connections.push_back(neighbor_id);
      auto &neighbor_level_connections = nodes_[neighbor_id].connections[level];
      neighbor_level_connections.push_back(current_id);
    }
  }

  std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>>
  searchLayer(const T *query, id_t entry_point_id, int level, size_t ef) const {
    std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>>
        candidates_min_heap;
    std::priority_queue<Neighbor> results_max_heap;
    std::unordered_set<id_t> visited;

    if (entry_point_id == INVALID_ID || entry_point_id >= nodes_.size()) {
      return candidates_min_heap;
    }

    if (nodes_[entry_point_id].level < level) {
      return candidates_min_heap;
    }

    float entry_dist = getDistance(query, entry_point_id);
    candidates_min_heap.emplace(entry_point_id, entry_dist);
    results_max_heap.emplace(entry_point_id, entry_dist);
    visited.insert(entry_point_id);

    while (!candidates_min_heap.empty()) {
      Neighbor current_best_candidate = candidates_min_heap.top();
      candidates_min_heap.pop();

      if (!results_max_heap.empty() &&
          current_best_candidate.distance > results_max_heap.top().distance &&
          results_max_heap.size() >= ef) {
        break;
      }

      if (current_best_candidate.id >= nodes_.size() ||
          nodes_[current_best_candidate.id].connections.size() <= level) {
        continue;
      }
      const auto &neighbors =
          nodes_[current_best_candidate.id].connections[level];

      for (id_t neighbor_id : neighbors) {
        if (visited.find(neighbor_id) == visited.end()) {
          visited.insert(neighbor_id);
          float dist = getDistance(query, neighbor_id);

          if (results_max_heap.size() < ef ||
              dist < results_max_heap.top().distance) {
            candidates_min_heap.emplace(neighbor_id, dist);
            results_max_heap.emplace(neighbor_id, dist);

            if (results_max_heap.size() > ef) {
              results_max_heap.pop();
            }
          }
        }
      }
    }

    std::priority_queue<Neighbor, std::vector<Neighbor>, std::greater<Neighbor>>
        final_results_min_heap;
    while (!results_max_heap.empty()) {
      final_results_min_heap.push(results_max_heap.top());
      results_max_heap.pop();
    }

    return final_results_min_heap;
  }

  int getRandomLevel(double reverse_size) {
    std::uniform_real_distribution<> distribution(0.0, 1.0);
    double r = -std::log(distribution(level_generator_)) * reverse_size;
    return static_cast<int>(r);
  }

private:
  int dimension_;
  std::size_t M_;
  std::size_t M_max_;
  std::size_t M_max0_;
  std::size_t ef_construction_;
  std::size_t ef_search_;

  double mult_;
  id_t entry_point_;
  int max_level_;
  std::atomic<id_t> element_count_{0};
  std::default_random_engine level_generator_;

  std::vector<InternalNode> nodes_;
  std::unordered_map<id_t, const T *> id_to_data_;
  std::unordered_map<label_t, id_t> label_to_id_;
  std::unordered_map<id_t, label_t> id_to_label_;

  DistanceAlgorithm<T> dist_algo_;
  SamplingAlgorithm<T, DistanceAlgorithm> sampler_;
};

}; // namespace bnsw