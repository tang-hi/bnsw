#include "adsampling.hpp"
#include "dist_alg/l2_distance.hpp"
#include "utils.hpp"
#include <algorithm>
#include <catch2/benchmark/catch_benchmark.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <limits>
#include <queue>
#include <random>
#include <spdlog/spdlog.h>
#include <utility>
#include <vector>

void runAdSamplingBenchmark(int dim, int topK,
                            const std::vector<std::vector<float>> &data,
                            const std::vector<float> &query,
                            const Eigen::MatrixXf &matrix) {
  std::priority_queue<std::pair<float, int>> pq;
  L2Distance<float> l2_distance;
  bnsw::AdSampling<float> ad_sampling(dim);
  ad_sampling.set_orthogonal_matrix(&matrix);
  float threshold = std::numeric_limits<float>::max();
  float distance = 0.0;
  for (auto i = 0U; i < data.size(); ++i) {
    if (!ad_sampling.above_threshold(data[i].data(), query.data(), threshold,
                                     distance)) {
      if (pq.size() < static_cast<uint32_t>(topK)) {
        pq.push({distance, i});

      } else {
        pq.pop();
        pq.push({distance, i});
        threshold = pq.top().first;
      }
    }
  }
}

void runFullScanBenchmark(int dim, int topK,
                          const std::vector<std::vector<float>> &data,
                          const std::vector<float> &query) {
  std::priority_queue<std::pair<float, int>> pq;
  L2Distance<float> l2_distance;

  for (auto i = 0U; i < data.size(); ++i) {

    float distance = l2_distance.distance(query.data(), data[i].data(), dim);
    if (pq.size() < static_cast<uint32_t>(topK)) {
      pq.push({distance, i});

    } else if (distance < pq.top().first) {
      pq.pop();
      pq.push({distance, i});
    }
  }
}

void setup(std::vector<std::vector<float>> &data, std::vector<float> &query,
           int dim, int data_size) {
  data.resize(data_size);
  std::for_each(data.begin(), data.end(),
                [&](auto &iter) { iter.resize(dim); });

  std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(-100.0, 100.0);
  // prepare data

  for (int i = 0; i < data_size; ++i) {
    for (int j = 0; j < dim; ++j) {
      data[i][j] = distribution(generator);
    }
  }

  // prepare query
  query.resize(dim);
  for (int i = 0; i < dim; ++i) {
    query[i] = distribution(generator);
  }
}

TEST_CASE("AdSampling Bench", "[benchmark]") {

  std::vector<std::vector<float>> data_128;
  std::vector<float> query_128;
  int data_size = 200000;
  int topK = 100;
  auto matrix_128 = createOrthogonal(128);
  setup(data_128, query_128, 128, data_size);
  BENCHMARK("AdSampling 128") {
    runAdSamplingBenchmark(128, topK, data_128, query_128, matrix_128);
  };

  std::vector<std::vector<float>> data_256;
  std::vector<float> query_256;
  setup(data_256, query_256, 256, data_size);
  auto matrix_256 = createOrthogonal(256);
  BENCHMARK("AdSampling 256") {
    runAdSamplingBenchmark(256, topK, data_256, query_256, matrix_256);
  };

  std::vector<std::vector<float>> data_512;
  std::vector<float> query_512;
  auto matrix_512 = createOrthogonal(512);
  setup(data_512, query_512, 512, data_size);

  BENCHMARK("AdSampling 512") {
    runAdSamplingBenchmark(512, topK, data_512, query_512, matrix_512);
  };

  std::vector<std::vector<float>> data_1024;
  std::vector<float> query_1024;
  auto matrix_1024 = createOrthogonal(1024);
  setup(data_1024, query_1024, 1024, data_size);
  BENCHMARK("AdSampling 1024") {
    runAdSamplingBenchmark(1024, topK, data_1024, query_1024, matrix_1024);
  };
}

TEST_CASE("Full Scan Bench", "[benchmark]") {
  std::vector<std::vector<float>> data_128;
  std::vector<float> query_128;
  int data_size = 200000;
  int topK = 100;
  setup(data_128, query_128, 128, data_size);
  BENCHMARK("Full Scan 128") {
    runFullScanBenchmark(128, topK, data_128, query_128);
  };

  std::vector<std::vector<float>> data_256;
  std::vector<float> query_256;
  setup(data_256, query_256, 256, data_size);
  BENCHMARK("Full Scan 256") {
    runFullScanBenchmark(256, topK, data_256, query_256);
  };

  std::vector<std::vector<float>> data_512;
  std::vector<float> query_512;
  setup(data_512, query_512, 512, data_size);
  BENCHMARK("Full Scan 512") {
    runFullScanBenchmark(512, topK, data_512, query_512);
  };

  std::vector<std::vector<float>> data_1024;
  std::vector<float> query_1024;
  setup(data_1024, query_1024, 1024, data_size);
  BENCHMARK("Full Scan 1024") {
    runFullScanBenchmark(1024, topK, data_1024, query_1024);
  };
}