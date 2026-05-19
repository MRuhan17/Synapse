#include "synapse/tensor.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>

using synapse::Tensor;

namespace {

void assert_true(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "Assertion failed: " << message << "\n";
        std::exit(1);
    }
}

} // namespace

int main() {
    Tensor x({128});
    for (size_t i = 0; i < 128; ++i) {
        x.set({i}, static_cast<double>(i) / 10.0);
    }
    x.set_requires_grad(true);

    Tensor out = x;
    for (size_t i = 0; i < 500; ++i) {
        out = out * 1.0001 + 0.001;
    }
    Tensor loss = synapse::sum(out);
    loss.backward();

    const auto& grad = x.grad();
    assert_true(grad.has_value(), "stress grad populated");
    assert_true(std::isfinite(grad->at({0})), "stress grad finite");

    std::cout << "Autograd stress test passed.\n";
    return 0;
}
