#include "adsampling.hpp"
#include "bnsw.hpp"
#include "dist_alg/l2_distance.hpp"
#include "nonsampling.hpp"
#include "spdlog/spdlog.h"
#include <Eigen/src/Core/Matrix.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unordered_set>
int main(int argc, char *argv[]) {

  int ef_search_value = 1500;
  if (argc == 2) {
    ef_search_value = std::stoi(argv[1]);
    if (ef_search_value <= 0) {
      std::cerr << "Error: ef_search_value must be a positive integer."
                << std::endl;
      return 1;
    }
  }

  if (argc != 7 && argc != 8) {

    spdlog::info("Usage: {} build <data_path> <index_path> <ef_construct> <M> "
                 "<dim> <sampling>",
                 argv[0]);
    spdlog::info(
        "Usage: {} search <index_path> <query_path> <gt_path> <k> <ef_search>  "
        "<dim> <sampling>",
        argv[0]);
  }

  std::string cmd = argv[1];
  if (cmd == "build") {
    spdlog::info("Building index...");
    std::filesystem::path data_path = argv[2];
    std::filesystem::path index_path = argv[3];
    int ef_construct = std::stoi(argv[4]);
    int M = std::stoi(argv[5]);
    int dim = std::stoi(argv[6]);
    int sampling = std::stoi(argv[7]);
    if (ef_construct <= 0 || M <= 0 || dim <= 0) {
      std::cerr << "Error: ef_construct, M, and dim must be positive integers."
                << std::endl;
      return 1;
    }
    std::ifstream data_file(data_path, std::ios::binary);
    if (!data_file) {
      spdlog::error("Error opening file: {}", data_path.string());
      return 1;
    }

    // Read the number of vectors and their dimension
    data_file.read(reinterpret_cast<char *>(&dim), sizeof(int));
    int num_vectors = 0;
    num_vectors = std::filesystem::file_size(data_path) / (dim * sizeof(float));

    // // Read the vectors
    std::vector<std::vector<float>> vectors(num_vectors,
                                            std::vector<float>(dim));
    data_file.seekg(0, std::ios::beg);
    for (int i = 0; i < num_vectors; ++i) {
      data_file.read(reinterpret_cast<char *>(&dim), sizeof(int));
      data_file.read(reinterpret_cast<char *>(vectors[i].data()),
                     dim * sizeof(float));
    }

    // // Close the file
    data_file.close();

    if (sampling == 0) {
      bnsw::bnsw<float, L2Distance, bnsw::NonSampling> bnsw_instance(
          dim, M, ef_construct, ef_search_value);
      auto build_start = std::chrono::high_resolution_clock::now();
      spdlog::info("Building BNSW index...");
      for (int i = 0; i < num_vectors; ++i) {
        bnsw_instance.addPoint(vectors[i].data(), i);
        if (i % 10000 == 0) {
          spdlog::info("Added {} vectors", i);
        }
      }
      auto build_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> build_duration = build_end - build_start;
      spdlog::info("BNSW index built in {} seconds, avg {} ms",
                   build_duration.count(),
                   build_duration.count() / num_vectors * 1000);
      bnsw_instance.saveIndex(index_path.string());

    } else {
      bnsw::bnsw<float, L2Distance, bnsw::AdSampling> bnsw_instance(
          dim, M, ef_construct, ef_search_value);
      auto build_start = std::chrono::high_resolution_clock::now();
      spdlog::info("Building BNSW index...");
      for (int i = 0; i < num_vectors; ++i) {
        bnsw_instance.addPoint(vectors[i].data(), i);
        if (i % 10000 == 0) {
          spdlog::info("Added {} vectors", i);
        }
      }
      auto build_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> build_duration = build_end - build_start;
      spdlog::info("BNSW index built in {} seconds, avg {} ms",
                   build_duration.count(),
                   build_duration.count() / num_vectors * 1000);
      bnsw_instance.saveIndex(index_path.string());
    }
  } else if (cmd == "search") {
    std::filesystem::path index_path = argv[2];
    std::filesystem::path query_path = argv[3];
    std::filesystem::path gt_path = argv[4];
    int k = std::stoi(argv[5]);
    int ef_search = std::stoi(argv[6]);
    int dim = std::stoi(argv[7]);
    int sampling = std::stoi(argv[8]);
    if (k <= 0 || ef_search <= 0 || dim <= 0) {
      std::cerr << "Error: k, ef_search, and dim must be positive integers."
                << std::endl;
      return 1;
    }
    std::ifstream gist1m_query_file(query_path, std::ios::binary);
    if (!gist1m_query_file) {
      spdlog::error("Error opening file: {}", query_path.string());
      return 1;
    }

    // Read the number of vectors and their dimension
    int num_query_vectors = 0;
    gist1m_query_file.read(reinterpret_cast<char *>(&dim), sizeof(int));
    num_query_vectors =
        std::filesystem::file_size(query_path) / (dim * sizeof(float));
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

    int num_gt_vectors = 0;
    std::ifstream gist1m_gt_file(gt_path, std::ios::binary);
    gist1m_gt_file.read(reinterpret_cast<char *>(&dim), sizeof(int));
    num_gt_vectors =
        std::filesystem::file_size(gt_path) / (k * sizeof(uint32_t));
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

    std::vector<std::vector<uint32_t>> results(num_query_vectors);
    int total_recall = 0;
    int early_stop = 0;
    int distance_calc_count = 0;
    if (sampling == 0) {
      bnsw::bnsw<float, L2Distance, bnsw::NonSampling> bnsw_instance(
          dim, 16, 500, ef_search_value);
      bnsw_instance.loadIndex(index_path.string());
      auto search_start = std::chrono::high_resolution_clock::now();
      spdlog::info("Searching for {} nearest neighbors...", k);
      for (int i = 0; i < num_query_vectors; ++i) {
        results[i] = bnsw_instance.search(query_vectors[i].data(), k);
        if (i % 1000 == 0) {
          spdlog::info("Processed {} query vectors", i);
        }
      }
      auto search_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> search_duration = search_end - search_start;
      early_stop = bnsw_instance.getEarlyStopCount();
      distance_calc_count = bnsw_instance.getDistanceCalcCount();
      spdlog::info("BNSW search completed in {} seconds, avg {} ms, avg dim "
                   "calculate {} ns",
                   search_duration.count(),
                   search_duration.count() / num_query_vectors * 1000,
                   search_duration.count() /
                       bnsw_instance.getDistanceCalcCount() * 1000 * 1000 *
                       1000);

    } else {
      bnsw::bnsw<float, L2Distance, bnsw::AdSampling> bnsw_instance(
          dim, 16, 500, ef_search_value);
      bnsw_instance.loadIndex(index_path.string());
      auto search_start = std::chrono::high_resolution_clock::now();
      spdlog::info("Searching for {} nearest neighbors...", k);
      for (int i = 0; i < num_query_vectors; ++i) {
        results[i] = bnsw_instance.search(query_vectors[i].data(), k);
        if (i % 1000 == 0) {
          spdlog::info("Processed {} query vectors", i);
        }
      }
      auto search_end = std::chrono::high_resolution_clock::now();
      std::chrono::duration<double> search_duration = search_end - search_start;
      early_stop = bnsw_instance.getEarlyStopCount();
      distance_calc_count = bnsw_instance.getDistanceCalcCount();
      spdlog::info("BNSW search completed in {} seconds, avg {} ms, avg dim "
                   "calculate {} ns",
                   search_duration.count(),
                   search_duration.count() / num_query_vectors * 1000,
                   search_duration.count() /
                       bnsw_instance.getDistanceCalcCount() * 1000 * 1000 *
                       1000);
    }
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
    spdlog::info("early stop: {}", early_stop);
    spdlog::info("distance calc count: {}", distance_calc_count);
  }
}