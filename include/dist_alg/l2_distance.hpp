#pragma once

#include <cstddef>
#include <cstdint>
#include <iostream>

template <typename T>
T L2SqrDistanceNaive(const T *a, const T *b, std::size_t dim) {
  T sum = 0;
  for (std::size_t i = 0; i < dim; ++i) {
    // std::cout << "query[" << i << "] = " << a[i] << std::endl;
    // std::cout << "V[" << i << "] = " << b[i] << std::endl;
    T diff = a[i] - b[i];
    // std::cout << "diff[" << i << "] = " << diff << std::endl;
    sum += diff * diff;
    // std::cout << "sum[" << i << "] = " << sum << std::endl;
  }
  // std::cout << "dim = " << dim  << " sum = " << sum << std::endl;
  return sum;
}

#ifdef __aarch64__

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

template <typename T>
T L2SqrDistanceNeon(const T *a, const T *b, std::size_t dim) {
  float32x4_t sum_vec = vdupq_n_f32(0.0f);
  std::size_t i = 0;
  for (; i + 4 < dim; i += 4) {
    float32x4_t va = vld1q_f32(a + i);
    float32x4_t vb = vld1q_f32(b + i);
    float32x4_t diff = vsubq_f32(va, vb);
    sum_vec = vmlaq_f32(sum_vec, diff, diff);
  }

  T temp[4] = {0};

  vst1q_f32(temp, sum_vec);

  T sum = temp[0] + temp[1] + temp[2] + temp[3];

  for (; i < dim; ++i) {
    T diff = a[i] - b[i];
    sum += diff * diff;
  }

  return sum;
}

#endif // __aarch64__

#ifdef __x86_64__
#include <immintrin.h>
template <typename T>
T L2SqrDistanceAVX2(const T *a, const T *b, std::size_t dim) {
  __m256 sum_vec = _mm256_setzero_ps();
  std::size_t i = 0;
  for (; i + 8 < dim; i += 8) {
    __m256 va = _mm256_loadu_ps(a + i);
    __m256 vb = _mm256_loadu_ps(b + i);
    __m256 diff = _mm256_sub_ps(va, vb);
    sum_vec = _mm256_fmadd_ps(diff, diff, sum_vec);
  }

  float temp[8] = {0};
  _mm256_storeu_ps(temp, sum_vec);

  T sum = temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6] +
          temp[7];

  for (; i < dim; ++i) {
    T diff = a[i] - b[i];
    sum += diff * diff;
  }

  return sum;
}

#endif

template <typename T> class L2Distance {
  using DistFunc = T (*)(const T *, const T *, std::size_t);

public:
  // Do runtime dispatching to select the best implementation
  L2Distance() {
#ifdef __aarch64__
    distance_function = L2SqrDistanceNeon<T>;
#elif defined(__x86_64__)
    distance_function = L2SqrDistanceAVX2<T>;
    // distance_function = L2SqrDistanceNaive<T>;
#else
    distance_function = L2SqrDistanceNaive<T>;
#endif
  }

  T distance(const T *a, const T *b, std::size_t dim) const {
    distance_calc_count += dim;
    return distance_function(a, b, dim);
  }

  uint64_t get_distance_calc_count() const {
    return distance_calc_count;
  }

private:
  static uint64_t distance_calc_count;
  static DistFunc distance_function;
};

template <typename T>
typename L2Distance<T>::DistFunc L2Distance<T>::distance_function = nullptr;

template <typename T>
uint64_t L2Distance<T>::distance_calc_count = 0U;