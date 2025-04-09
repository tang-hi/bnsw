#pragma once

#include <cmath>
#include <cstddef>

template <typename T>
T L2DistanceNaive(const T *a, const T *b, std::size_t dim) {
  T sum = 0;
  for (std::size_t i = 0; i < dim; ++i) {
    T diff = a[i] - b[i];
    sum += diff * diff;
  }
  return std::sqrt(sum);
}

#ifdef __aarch64__

#include <arm_neon.h>
template <typename T>
T L2DistanceNeon(const T *a, const T *b, std::size_t dim) {
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

  return std::sqrt(sum);
}

#endif // __aarch64__

#ifdef __x86_64__
#include <immintrin.h>
template <typename T>
T L2DistanceAVX2(const T *a, const T *b, std::size_t dim) {
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

  return std::sqrt(sum);
}

#endif

template <typename T> class L2Distance {
  using DistFunc = T (*)(const T *, const T *, std::size_t);

public:
  // Do runtime dispatching to select the best implementation
  L2Distance() {
#ifdef __aarch64__
    distance_function = L2DistanceNeon<T>;
#elif defined(__x86_64__)
    distance_function = L2DistanceAVX2<T>;
#else
    distance_function = L2DistanceNaive<T>;
#endif
  }

  T distance(const T *a, const T *b, std::size_t dim) {
    return distance_function(a, b, dim);
  }

private:
  static DistFunc distance_function;
};