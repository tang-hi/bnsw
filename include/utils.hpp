#pragma once
#include "Eigen/Dense"

inline Eigen::MatrixXf createOrthogonal(int dim) {
  auto orthogonal = Eigen::MatrixXf::Random(dim, dim);
  Eigen::HouseholderQR<Eigen::MatrixXf> qr(orthogonal);
  return qr.householderQ();
}