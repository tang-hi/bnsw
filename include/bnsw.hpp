#pragma once

#include "adsampling.hpp"
#include "utils.hpp"
#include <Eigen/src/Core/Matrix.h>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace bnsw {

struct BnswTestAccessor;

template <typename T, template <typename U> class DistanceAlgorithm,
          template <typename V,
                    template <typename> typename> class SamplingAlgorithm =
              AdSampling>
class bnsw {
  friend struct BnswTestAccessor;
  using id_t = std::uint32_t;
  using label_t = std::uint32_t;

  static constexpr id_t INVALID_ID = std::numeric_limits<id_t>::max();
  static constexpr bool need_convert =
      SamplingAlgorithm<T, DistanceAlgorithm>::need_convert;

public:
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
        mult_(1.0 / std::log(1.0 * M)), sampler_(dimension),
        orthogonal_matrix_(createOrthogonal(dimension)) {

    static_assert(std::is_default_constructible_v<DistanceAlgorithm<T>>,
                  "DistanceAlgorithm must be default constructible");
    if (M == 0)
      throw std::invalid_argument("M cannot be 0");
    if (ef_construction == 0)
      throw std::invalid_argument("ef_construction cannot be 0");
    sampler_.set_orthogonal_matrix(&orthogonal_matrix_);
  }

  bnsw() = default;

  ~bnsw() = default;

  bool addPoint(const void *point, label_t label, int level) {
    const T *point_data = static_cast<const T *>(point);
    if constexpr (need_convert) {
      sampler_.convert(point_data);
    }
    id_t current_id;
    int current_level;

    if (label_to_id_.count(label)) {
      // duplicate label
      return false;
    }

    current_id = element_count_++;
    label_to_id_[label] = current_id;
    id_to_label_[current_id] = label;
    id_to_data_[current_id] = point_data;

    if (level == -1) {
      current_level = getRandomLevel(mult_);
    } else {
      // friendly to unit test
      current_level = level;
    }

    nodes_.emplace_back(current_level, point_data, M_max_, M_max0_);

    id_t current_entry_point = entry_point_;
    int current_max_level = max_level_;

    // empty graph
    if (current_entry_point == INVALID_ID) {
      entry_point_ = current_id;
      max_level_ = current_level;
      return true;
    }

    id_t nearest_node = current_entry_point;
    for (int level = current_max_level; level > current_level; --level) {
      nearest_node = searchLayer(point_data, nearest_node, level, 1).top().id;
    }

    for (int level = std::min(current_level, current_max_level); level >= 0;
         --level) {
      auto neighbors_min_heap =
          searchLayer(point_data, nearest_node, level, ef_construction_);
      size_t M_level = (level == 0) ? M_max0_ : M_max_;
      nearest_node = selectAndConnectNeighbors(current_id, neighbors_min_heap,
                                               M_level, level);
    }

    // update entry point if necessary eg. node's level is higher than current
    // max level
    if (current_level > current_max_level) {
      max_level_ = current_level;
      entry_point_ = current_id;
    }
    return true;
  }

  bool addPoint(const void *point, label_t label) {
    return addPoint(point, label, -1);
  }

  auto search(const void *query, std::size_t k) const -> std::vector<label_t> {
    const T *query_typed = static_cast<const T *>(query);
    if constexpr (need_convert) {
      sampler_.convert(query_typed);
    }
    id_t current_entry_point = entry_point_;
    int current_max_level = max_level_;

    if (current_entry_point == INVALID_ID || nodes_.empty()) {
      return {};
    }

    id_t nearest_node = current_entry_point;
    float min_distance = getDistance(query_typed, nearest_node);
    for (int level = current_max_level; level > 0; --level) {
      bool changed = true;
      while (changed) {
        changed = false;
        const auto &neighbors = nodes_[nearest_node].connections[level];
        for (id_t neighbor_id : neighbors) {
          if (neighbor_id >= nodes_.size()) {
            throw std::runtime_error("Invalid neighbor ID");
          }
          float dist = getDistance(query_typed, neighbor_id);
          if (dist < min_distance) {
            min_distance = dist;
            nearest_node = neighbor_id;
            changed = true;
          }
        }
      }
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
    return dist_algo_.distance(data1, data2, dimension_);
  }

  float getDistance(const T *query, id_t id) const {
    const T *data = id_to_data_.at(id);
    if (!data || !query) {
      throw std::runtime_error("Invalid data pointer encountered");
    }
    return dist_algo_.distance(query, data, dimension_);
  }

  /**
   * @brief Prune the neighbors in the candidates_min_heap to keep at most
   * limits neighbors
   *
   * @param candidates_min_heap the candidates to prune
   * @param limits the limits to keep
   */
  void pruneNeighbors(
      std::priority_queue<Neighbor, std::vector<Neighbor>,
                          std::greater<Neighbor>> &candidates_min_heap,
      const uint32_t limits) {
    // no need to prune
    if (candidates_min_heap.size() < limits) {
      return;
    }

    std::vector<Neighbor> top_candidates;

    while (candidates_min_heap.size()) {
      if (top_candidates.size() == limits) {
        break;
      }
      auto curent_pair = candidates_min_heap.top();
      candidates_min_heap.pop();
      bool good = true;

      for (auto &neighbor : top_candidates) {
        auto curdist = getDistance(curent_pair.id, neighbor.id);
        if (curdist < curent_pair.distance) {
          good = false;
          break;
        }
      }
      if (good) {
        top_candidates.push_back(curent_pair);
      }
    }
    candidates_min_heap = {};
    for (const auto &candidate : top_candidates) {
      candidates_min_heap.emplace(candidate);
    }
    return;
  }

  id_t selectAndConnectNeighbors(
      id_t current_id,
      std::priority_queue<Neighbor, std::vector<Neighbor>,
                          std::greater<Neighbor>> &candidates_min_heap,
      size_t M, int level) {
    pruneNeighbors(candidates_min_heap, M_);
    auto &current_connections = nodes_[current_id].connections[level];
    current_connections.reserve(M);
    auto nearst_node = candidates_min_heap.top().id;

    while (!candidates_min_heap.empty() && current_connections.size() < M) {
      id_t neighbor_id = candidates_min_heap.top().id;
      candidates_min_heap.pop();

      current_connections.push_back(neighbor_id);
      auto &neighbor_level_connections = nodes_[neighbor_id].connections[level];
      neighbor_level_connections.push_back(current_id);
      if (neighbor_level_connections.size() > M) {
        std::priority_queue<Neighbor, std::vector<Neighbor>,
                            std::greater<Neighbor>>
            candidates;
        for (auto &id : neighbor_level_connections) {
          candidates.emplace(id, getDistance(neighbor_id, id));
        }
        pruneNeighbors(candidates, M);
        neighbor_level_connections.clear();
        while (!candidates.empty()) {
          neighbor_level_connections.push_back(candidates.top().id);
          candidates.pop();
        }
      }
    }

    return nearst_node;
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

      const auto &neighbors =
          nodes_[current_best_candidate.id].connections[level];

      float lower_bound = results_max_heap.top().distance;
      for (id_t neighbor_id : neighbors) {
        if (visited.find(neighbor_id) == visited.end()) {
          visited.insert(neighbor_id);
          if (results_max_heap.size() < ef) {
            float dist = getDistance(query, neighbor_id);
            candidates_min_heap.emplace(neighbor_id, dist);
            results_max_heap.emplace(neighbor_id, dist);
            lower_bound = results_max_heap.top().distance;
          } else if (float dist = 0.0; !sampler_.above_threshold(
                         query, id_to_data_.at(neighbor_id), lower_bound,
                         dist)) {
            candidates_min_heap.emplace(neighbor_id, dist);
            results_max_heap.emplace(neighbor_id, dist);
            lower_bound = results_max_heap.top().distance;
          }

          if (results_max_heap.size() > ef) {
            results_max_heap.pop();
          }
          // if (results_max_heap.size() < ef ||
          //     dist < results_max_heap.top().distance) {
          //   candidates_min_heap.emplace(neighbor_id, dist);
          //   results_max_heap.emplace(neighbor_id, dist);

          //   if (results_max_heap.size() > ef) {
          //     results_max_heap.pop();
          //   }
          // }
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

  void saveIndex(const std::string &location) {
    std::ofstream ofs(location, std::ios::binary);
    if (!ofs) {
      throw std::runtime_error("Failed to open file for writing");
    }
    ofs.write(reinterpret_cast<const char *>(&dimension_), sizeof(dimension_));
    ofs.write(reinterpret_cast<const char *>(&M_), sizeof(M_));
    ofs.write(reinterpret_cast<const char *>(&M_max_), sizeof(M_max_));
    ofs.write(reinterpret_cast<const char *>(&M_max0_), sizeof(M_max0_));
    ofs.write(reinterpret_cast<const char *>(&ef_construction_),
              sizeof(ef_construction_));
    ofs.write(reinterpret_cast<const char *>(&ef_search_), sizeof(ef_search_));
    ofs.write(reinterpret_cast<const char *>(&entry_point_),
              sizeof(entry_point_));
    ofs.write(reinterpret_cast<const char *>(&max_level_), sizeof(max_level_));
    ofs.write(reinterpret_cast<const char *>(&mult_), sizeof(mult_));
    ofs.write(reinterpret_cast<const char *>(&element_count_),
              sizeof(element_count_));
    for (const auto &node : nodes_) {
      int level = node.level;
      ofs.write(reinterpret_cast<const char *>(&level), sizeof(level));
      ofs.write(reinterpret_cast<const char *>(node.point_data),
                dimension_ * sizeof(T));
      std::size_t size = node.connections.size();
      ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));
      for (const auto &connections : node.connections) {
        std::size_t size = connections.size();
        ofs.write(reinterpret_cast<const char *>(&size), sizeof(size));
        ofs.write(reinterpret_cast<const char *>(connections.data()),
                  size * sizeof(id_t));
      }
    }
    std::size_t id_to_data_size = id_to_data_.size();
    ofs.write(reinterpret_cast<const char *>(&id_to_data_size),
              sizeof(id_to_data_size));
    for (const auto &[id, data] : id_to_data_) {
      ofs.write(reinterpret_cast<const char *>(&id), sizeof(id));
      ofs.write(reinterpret_cast<const char *>(data), dimension_ * sizeof(T));
    }
    std::size_t label_to_id_size = label_to_id_.size();
    ofs.write(reinterpret_cast<const char *>(&label_to_id_size),
              sizeof(label_to_id_size));
    for (const auto &[label, id] : label_to_id_) {
      ofs.write(reinterpret_cast<const char *>(&label), sizeof(label));
      ofs.write(reinterpret_cast<const char *>(&id), sizeof(id));
    }
    std::size_t id_to_label_size = id_to_label_.size();
    ofs.write(reinterpret_cast<const char *>(&id_to_label_size),
              sizeof(id_to_label_size));
    for (const auto &[id, label] : id_to_label_) {
      ofs.write(reinterpret_cast<const char *>(&id), sizeof(id));
      ofs.write(reinterpret_cast<const char *>(&label), sizeof(label));
    }

    std::size_t orthogonal_matrix_size = orthogonal_matrix_.size();
    ofs.write(reinterpret_cast<const char *>(&orthogonal_matrix_size),
              sizeof(orthogonal_matrix_size));
    ofs.write(reinterpret_cast<const char *>(orthogonal_matrix_.data()),
              orthogonal_matrix_size * sizeof(float));
    ofs.close();
  }

  void loadIndex(const std::string &location) {
    std::ifstream ifs(location, std::ios::binary);
    if (!ifs) {
      throw std::runtime_error("Failed to open file for reading");
    }
    ifs.read(reinterpret_cast<char *>(&dimension_), sizeof(dimension_));
    ifs.read(reinterpret_cast<char *>(&M_), sizeof(M_));
    ifs.read(reinterpret_cast<char *>(&M_max_), sizeof(M_max_));
    ifs.read(reinterpret_cast<char *>(&M_max0_), sizeof(M_max0_));
    ifs.read(reinterpret_cast<char *>(&ef_construction_),
             sizeof(ef_construction_));
    ifs.read(reinterpret_cast<char *>(&ef_search_), sizeof(ef_search_));
    ifs.read(reinterpret_cast<char *>(&entry_point_), sizeof(entry_point_));
    ifs.read(reinterpret_cast<char *>(&max_level_), sizeof(max_level_));
    ifs.read(reinterpret_cast<char *>(&mult_), sizeof(mult_));
    ifs.read(reinterpret_cast<char *>(&element_count_), sizeof(element_count_));
    nodes_.clear();
    nodes_.reserve(element_count_);
    for (std::size_t i = 0; i < element_count_; ++i) {
      int level;
      ifs.read(reinterpret_cast<char *>(&level), sizeof(level));
      T *point_data = new T[dimension_];
      ifs.read(reinterpret_cast<char *>(point_data), dimension_ * sizeof(T));
      InternalNode node(level, point_data, M_max_, M_max0_);
      std::size_t size;
      ifs.read(reinterpret_cast<char *>(&size), sizeof(size));
      node.connections.resize(size);
      for (auto &connections : node.connections) {
        std::size_t size;
        ifs.read(reinterpret_cast<char *>(&size), sizeof(size));
        connections.resize(size);
        ifs.read(reinterpret_cast<char *>(connections.data()),
                 size * sizeof(id_t));
      }
      nodes_.push_back(node);
    }

    std::size_t id_to_data_size;
    ifs.read(reinterpret_cast<char *>(&id_to_data_size),
             sizeof(id_to_data_size));
    id_to_data_.clear();
    id_to_data_.reserve(id_to_data_size);
    for (std::size_t i = 0; i < id_to_data_size; ++i) {
      id_t id;
      ifs.read(reinterpret_cast<char *>(&id), sizeof(id));
      T *data = new T[dimension_];
      ifs.read(reinterpret_cast<char *>(data), dimension_ * sizeof(T));
      id_to_data_[id] = data;
    }

    std::size_t label_to_id_size;
    ifs.read(reinterpret_cast<char *>(&label_to_id_size),
             sizeof(label_to_id_size));
    label_to_id_.clear();
    label_to_id_.reserve(label_to_id_size);
    for (std::size_t i = 0; i < label_to_id_size; ++i) {
      label_t label;
      ifs.read(reinterpret_cast<char *>(&label), sizeof(label));
      id_t id;
      ifs.read(reinterpret_cast<char *>(&id), sizeof(id));
      label_to_id_[label] = id;
    }
    std::size_t id_to_label_size;
    ifs.read(reinterpret_cast<char *>(&id_to_label_size),
             sizeof(id_to_label_size));
    id_to_label_.clear();
    id_to_label_.reserve(id_to_label_size);
    for (std::size_t i = 0; i < id_to_label_size; ++i) {
      id_t id;
      ifs.read(reinterpret_cast<char *>(&id), sizeof(id));
      label_t label;
      ifs.read(reinterpret_cast<char *>(&label), sizeof(label));
      id_to_label_[id] = label;
    }
    std::size_t orthogonal_matrix_size;
    ifs.read(reinterpret_cast<char *>(&orthogonal_matrix_size),
             sizeof(orthogonal_matrix_size));
    orthogonal_matrix_.resize(orthogonal_matrix_size);
    ifs.read(reinterpret_cast<char *>(orthogonal_matrix_.data()),
             orthogonal_matrix_size * sizeof(float));
    sampler_.set_orthogonal_matrix(&orthogonal_matrix_);
    ifs.close();
  }

private:
  int dimension_;
  std::size_t M_;
  std::size_t M_max_;
  std::size_t M_max0_;
  std::size_t ef_construction_;
  std::size_t ef_search_;

  std::default_random_engine level_generator_;
  id_t entry_point_;
  int max_level_;
  double mult_;
  std::atomic<id_t> element_count_{0};

  std::vector<InternalNode> nodes_;
  std::unordered_map<id_t, const T *> id_to_data_;
  std::unordered_map<label_t, id_t> label_to_id_;
  std::unordered_map<id_t, label_t> id_to_label_;

  DistanceAlgorithm<T> dist_algo_{};
  SamplingAlgorithm<T, DistanceAlgorithm> sampler_;
  Eigen::MatrixXf orthogonal_matrix_;
};

}; // namespace bnsw