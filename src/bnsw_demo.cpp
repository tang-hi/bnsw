#include "bnsw.hpp"
#include "dist_alg/l2_distance.hpp"
#include <filesystem>
#include <iostream>

int main(int argc, char **argv) {

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " dataset location" << std::endl;
    return 1;
  }

  char *dataset_location = argv[1];

  // read dataset
  std::filesystem::path dataset_path(dataset_location);
  if (!std::filesystem::exists(dataset_path)) {
    std::cerr << "Dataset location does not exist: " << dataset_location
              << std::endl;
    return 1;
  }

  // test recall
  return 0;
}