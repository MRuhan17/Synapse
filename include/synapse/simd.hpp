#pragma once

#include <cstddef>

namespace synapse::simd {

void add(const float* a, const float* b, float* out, size_t count);
void add(const double* a, const double* b, double* out, size_t count);
void sub(const float* a, const float* b, float* out, size_t count);
void sub(const double* a, const double* b, double* out, size_t count);
void mul(const float* a, const float* b, float* out, size_t count);
void mul(const double* a, const double* b, double* out, size_t count);
void div(const float* a, const float* b, float* out, size_t count);
void div(const double* a, const double* b, double* out, size_t count);

void add_scalar(const float* a, float scalar, float* out, size_t count);
void add_scalar(const double* a, double scalar, double* out, size_t count);
void mul_scalar(const float* a, float scalar, float* out, size_t count);
void mul_scalar(const double* a, double scalar, double* out, size_t count);

} // namespace synapse::simd
