#include "synapse/simd.hpp"

namespace synapse::simd {

void add(const float* a, const float* b, float* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] + b[i];
    }
}

void add(const double* a, const double* b, double* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] + b[i];
    }
}

void sub(const float* a, const float* b, float* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] - b[i];
    }
}

void sub(const double* a, const double* b, double* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] - b[i];
    }
}

void mul(const float* a, const float* b, float* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] * b[i];
    }
}

void mul(const double* a, const double* b, double* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] * b[i];
    }
}

void div(const float* a, const float* b, float* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] / b[i];
    }
}

void div(const double* a, const double* b, double* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] / b[i];
    }
}

void add_scalar(const float* a, float scalar, float* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] + scalar;
    }
}

void add_scalar(const double* a, double scalar, double* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] + scalar;
    }
}

void mul_scalar(const float* a, float scalar, float* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] * scalar;
    }
}

void mul_scalar(const double* a, double scalar, double* out, size_t count) {
    for (size_t i = 0; i < count; ++i) {
        out[i] = a[i] * scalar;
    }
}

} // namespace synapse::simd
