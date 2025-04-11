#pragma once

#include "adsampling.hpp"
#include <atomic>
#include <random>
namespace bnsw {

template <typename T, template <typename U> class DistanceAlgorithm,
          template <typename V,
                    template <typename> typename> class SamplingAlgorithm =
              AdSampling>
class bnsw {
  // id_t is the type used for internal node IDs
  using id_t = std::uint32_t;

  // label_t is the type used for external node IDs
  using label_t = std::uint32_t;

public:
  explicit bnsw(int dimension) : dimension_(dimension), sampler_(dimension) {}
  ~bnsw() = default;

  void addPoint(const void *point, label_t label) {
    int current_level = getRandomLevel(mult_);
    id_t point_id = element_count_++;
    if (entry_point_ == -1) {
      entry_point_ = point_id;
      max_level_ = current_level;
      return;
    } 

    

  }

  auto search(const void *query, int k) const -> std::vector<label_t> {
    std::vector<label_t> results;
    return results;
  }

private:
  int getRandomLevel(double reverse_size) const {
    std::uniform_real_distribution<> distribution(0.0, 1.0);
    double r = -log(distribution(level_generator_)) * reverse_size;
    return r;
  }

private:
  int dimension_;
  int max_level_;
  std::size_t M_;
  std::size_t ef_construction_;
  std::size_t ef_;

  double mult_{0.0};
  id_t entry_point_;
  mutable std::atomic<id_t> element_count_{0};
  std::default_random_engine level_generator_;
  SamplingAlgorithm<T, DistanceAlgorithm> sampler_;
};

}; // namespace bnsw