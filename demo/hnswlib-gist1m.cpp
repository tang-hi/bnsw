#include "hnswlib/hnswlib.h"
#include "spdlog/spdlog.h"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
int main() {

  // read gist1m
  std::filesystem::path root_path = "/home/hayes/projects/bnsw";
  std::filesystem::path gist1m_path =
      root_path / "dataset/gist/gist_base.fvecs";
  std::ifstream gist1m_file(gist1m_path, std::ios::binary);
  if (!gist1m_file) {
    spdlog::error("Error opening file: {}", gist1m_path.string());
    return 1;
  }

  // Read the number of vectors and their dimension
  int dim = 0;
  gist1m_file.read(reinterpret_cast<char *>(&dim), sizeof(int));
  int num_vectors = 0;
  num_vectors = std::filesystem::file_size(gist1m_path) / (dim * sizeof(float));

  // Read the vectors
  std::vector<std::vector<float>> vectors(num_vectors, std::vector<float>(dim));
  gist1m_file.seekg(0, std::ios::beg);
  for (int i = 0; i < num_vectors; ++i) {
    gist1m_file.read(reinterpret_cast<char *>(&dim), sizeof(int));
    gist1m_file.read(reinterpret_cast<char *>(vectors[i].data()),
                     dim * sizeof(float));
  }

  // Close the file
  gist1m_file.close();
  spdlog::info("Read {} vectors of dimension {} from {}", num_vectors, dim,
               gist1m_path.string());
  hnswlib::L2Space space(dim);
  hnswlib::HierarchicalNSW<float> hnsw(&space, num_vectors, 24, 500);
  auto build_start = std::chrono::high_resolution_clock::now();
  spdlog::info("Building BNSW index...");
  for (int i = 0; i < num_vectors; ++i) {
    hnsw.addPoint(vectors[i].data(), i);
    if (i % 10000 == 0) {
      spdlog::info("Added {} vectors", i);
    }
  }
  std::filesystem::path save_path = root_path / "dataset/gist/hnsw_index.bin";
  hnsw.saveIndex(save_path.string());

  auto build_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> build_duration = build_end - build_start;
  spdlog::info("HNSW index built in {} seconds, avg {} ms",
               build_duration.count(),
               build_duration.count() / num_vectors * 1000);

  // read gist1m query
  std::filesystem::path gist1m_query_path =
      root_path / "dataset/gist/gist_query.fvecs";
  std::ifstream gist1m_query_file(gist1m_query_path, std::ios::binary);
  if (!gist1m_query_file) {
    spdlog::error("Error opening file: {}", gist1m_query_path.string());
    return 1;
  }

  // Read the number of vectors and their dimension
  int num_query_vectors = 0;
  gist1m_query_file.read(reinterpret_cast<char *>(&dim), sizeof(int));
  num_query_vectors =
      std::filesystem::file_size(gist1m_query_path) / (dim * sizeof(float));
  // Read the vectors
  std::vector<std::vector<float>> query_vectors(num_query_vectors,
                                                std::vector<float>(dim));
  gist1m_query_file.seekg(0, std::ios::beg);
  for (int i = 0; i < num_query_vectors; ++i) {
    gist1m_query_file.read(reinterpret_cast<char *>(&dim), sizeof(int));
    gist1m_query_file.read(reinterpret_cast<char *>(query_vectors[i].data()),
                           dim * sizeof(float));
  }
  // Close the file
  gist1m_query_file.close();
  // Search for the nearest neighbors
  int k = 100;
  auto search_start = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<uint32_t>> results(num_query_vectors);
  for (int i = 0; i < num_query_vectors; ++i) {
    results[i].reserve(k);
  }
  spdlog::info("Searching for {} nearest neighbors...", k);
  for (int i = 0; i < num_query_vectors; ++i) {
    auto result = hnsw.searchKnn(query_vectors[i].data(), k);
    while (!result.empty()) {
      auto top = result.top();
      results[i].push_back(top.second);
      result.pop();
    }
    if (i % 1000 == 0) {
      spdlog::info("Processed {} query vectors", i);
    }
  }
  auto search_end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> search_duration = search_end - search_start;
  spdlog::info("BNSW search completed in {} seconds, avg {} ms",
               search_duration.count(),
               search_duration.count() / num_query_vectors * 1000);

  // Read the ground truth
  std::filesystem::path gist1m_gt_path =
      root_path / "dataset/gist/gist_groundtruth.ivecs";
  std::ifstream gist1m_gt_file(gist1m_gt_path, std::ios::binary);
  if (!gist1m_gt_file) {
    spdlog::error("Error opening file: {}", gist1m_gt_path.string());
    return 1;
  }
  // Read the number of vectors and their dimension
  int num_gt_vectors = 0;
  gist1m_gt_file.read(reinterpret_cast<char *>(&dim), sizeof(int));
  num_gt_vectors =
      std::filesystem::file_size(gist1m_gt_path) / (k * sizeof(uint32_t));
  gist1m_gt_file.seekg(0, std::ios::beg);
  // Read the vectors
  std::vector<std::vector<uint32_t>> gt_vectors(num_gt_vectors,
                                                std::vector<uint32_t>(k));
  for (int i = 0; i < num_gt_vectors; ++i) {
    gist1m_gt_file.read(reinterpret_cast<char *>(&dim), sizeof(int));
    gist1m_gt_file.read(reinterpret_cast<char *>(gt_vectors[i].data()),
                        k * sizeof(uint32_t));
  }

  gist1m_gt_file.close();

  // Calculate recall
  int total_recall = 0;
  for (int i = 0; i < num_query_vectors; ++i) {
    std::unordered_set<uint32_t> gt_set(gt_vectors[i].begin(),
                                        gt_vectors[i].end());
    int recall_count = 0;
    for (const auto &result : results[i]) {
      if (gt_set.find(result) != gt_set.end()) {
        recall_count++;
      }
    }
    total_recall += recall_count;
  }
  double recall_rate =
      static_cast<double>(total_recall) / (num_query_vectors * k);
  spdlog::info("Recall rate: {:.2f}%", recall_rate * 100);
}