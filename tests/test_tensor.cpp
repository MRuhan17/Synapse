#include "synapse/tensor.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>

using synapse::Slice;
using synapse::Tensor;

namespace {

void assert_true(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "Assertion failed: " << message << "\n";
        std::exit(1);
    }
}

void assert_near(double lhs, double rhs, double tol, const char* message) {
    if (std::fabs(lhs - rhs) > tol) {
        std::cerr << "Assertion failed: " << message << " (" << lhs << " vs " << rhs << ")\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    Tensor tensor({2, 3});
    double value = 1.0;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            tensor.set({i, j}, value);
            value += 1.0;
        }
    }

    assert_true(tensor.shape() == Tensor::Shape({2, 3}), "shape initialized");
    assert_true(tensor.strides() == Tensor::Strides({3, 1}), "strides initialized");
    assert_near(tensor.at({1, 2}), 6.0, 1e-9, "indexing works");

    Tensor reshaped = tensor.reshape({3, 2});
    assert_true(reshaped.shape() == Tensor::Shape({3, 2}), "reshape updates shape");
    assert_near(reshaped.at({2, 1}), 6.0, 1e-9, "reshape preserves data");

    Tensor transposed = tensor.transpose({1, 0});
    assert_true(transposed.shape() == Tensor::Shape({3, 2}), "transpose shape");
    assert_near(transposed.at({2, 1}), 6.0, 1e-9, "transpose indexing");

    Tensor sliced = tensor.slice({Slice{0, 2, 1}, Slice{1, 3, 1}});
    assert_true(sliced.shape() == Tensor::Shape({2, 2}), "slice shape");
    assert_near(sliced.at({0, 0}), 2.0, 1e-9, "slice value");

    Tensor vector({3});
    vector.set({0}, 1.0);
    vector.set({1}, 2.0);
    vector.set({2}, 3.0);
    Tensor broadcast_sum = tensor + vector;
    assert_near(broadcast_sum.at({0, 0}), 2.0, 1e-9, "broadcast add value");
    assert_near(broadcast_sum.at({1, 2}), 9.0, 1e-9, "broadcast add value 2");

    Tensor scaled = tensor * 2.0;
    assert_near(scaled.at({0, 2}), 6.0, 1e-9, "scalar multiply");

    Tensor left({2, 3});
    Tensor right({3, 2});
    double fill = 1.0;
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            left.set({i, j}, fill++);
        }
    }
    fill = 1.0;
    for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 2; ++j) {
            right.set({i, j}, fill++);
        }
    }
    Tensor product = synapse::matmul(left, right);
    assert_true(product.shape() == Tensor::Shape({2, 2}), "matmul shape");
    assert_near(product.at({0, 0}), 22.0, 1e-9, "matmul value");
    assert_near(product.at({1, 1}), 139.0, 1e-9, "matmul value 2");

    Tensor total = synapse::sum(tensor);
    assert_near(total.at({}), 21.0, 1e-9, "sum reduction");

    Tensor axis_sum = synapse::sum(tensor, 0);
    assert_true(axis_sum.shape() == Tensor::Shape({3}), "axis sum shape");
    assert_near(axis_sum.at({0}), 5.0, 1e-9, "axis sum value");

    Tensor mean = synapse::mean(tensor);
    assert_near(mean.at({}), 3.5, 1e-9, "mean reduction");

    Tensor max_val = synapse::max(tensor);
    assert_near(max_val.at({}), 6.0, 1e-9, "max reduction");

    Tensor view = tensor.view();
    view.set({0, 0}, 42.0);
    assert_near(tensor.at({0, 0}), 42.0, 1e-9, "view shares data");

    Tensor cloned = tensor.clone();
    cloned.set({0, 0}, -1.0);
    assert_near(tensor.at({0, 0}), 42.0, 1e-9, "clone is independent");

    std::vector<uint8_t> encoded = tensor.serialize();
    Tensor decoded = Tensor::deserialize(encoded);
    assert_near(decoded.at({1, 2}), 6.0, 1e-9, "serialization round trip");

    std::cout << "All tensor tests passed.\n";
    return 0;
}
