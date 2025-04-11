#pragma once
#include <queue>
namespace bnsw {
template <typename T, bool Negative = false> class Collector {

public:
  explicit Collector(int k) : k_{k} {}

  void collect(float score, T id) {
    if constexpr (Negative) {
      score = -score;
    }
    pq.emplace(score, id);
    if (pq.size() > k_) {
      pq.pop();
    }
  }

  float topScore() const {
    if (pq.empty()) {
      return 0.0f;
    }
    return pq.top().first;
  }

  auto size() const { return pq.size(); }
  auto empty() const { return pq.empty(); }

private:
  std::priority_queue<std::pair<float, T>> pq;
  int k_{0};
};

} // namespace bnsw