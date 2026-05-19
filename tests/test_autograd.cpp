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

void assert_near(double lhs, double rhs, double tol, const char* message) {
    if (std::fabs(lhs - rhs) > tol) {
        std::cerr << "Assertion failed: " << message << " (" << lhs << " vs " << rhs << ")\n";
        std::exit(1);
    }
}

Tensor make_sequential_values_tensor(const Tensor::Shape& shape, double start) {
    Tensor out(shape);
    double value = start;
    std::vector<size_t> index(shape.size(), 0);
    for (size_t i = 0; i < out.numel(); ++i) {
        out.set(index, value);
        value += 1.0;
        if (!shape.empty()) {
            for (size_t d = shape.size(); d-- > 0;) {
                index[d] += 1;
                if (index[d] < shape[d]) {
                    break;
                }
                index[d] = 0;
            }
        }
    }
    return out;
}

} // namespace

int main() {
    Tensor x({2, 2});
    x.set({0, 0}, 1.0);
    x.set({0, 1}, 2.0);
    x.set({1, 0}, 3.0);
    x.set({1, 1}, 4.0);
    x.set_requires_grad(true);

    Tensor y = x * 2.0 + 1.0;
    Tensor loss = synapse::sum(y);
    loss.backward();

    const auto& grad_x = x.grad();
    assert_true(grad_x.has_value(), "grad populated");
    assert_near(grad_x->at({0, 0}), 2.0, 1e-9, "grad value");
    assert_near(grad_x->at({1, 1}), 2.0, 1e-9, "grad value 2");

    Tensor a = make_sequential_values_tensor({2, 3}, 1.0);
    Tensor b = make_sequential_values_tensor({3}, 1.0);
    a.set_requires_grad(true);
    b.set_requires_grad(true);
    Tensor broadcast = a + b;
    synapse::sum(broadcast).backward();
    assert_near(a.grad()->at({0, 0}), 1.0, 1e-9, "broadcast grad a");
    assert_near(b.grad()->at({0}), 2.0, 1e-9, "broadcast grad b");
    assert_near(b.grad()->at({2}), 2.0, 1e-9, "broadcast grad b 2");

    Tensor lhs({2, 2});
    lhs.set({0, 0}, 1.0);
    lhs.set({0, 1}, 2.0);
    lhs.set({1, 0}, 3.0);
    lhs.set({1, 1}, 4.0);
    Tensor rhs({2, 2});
    rhs.set({0, 0}, 5.0);
    rhs.set({0, 1}, 6.0);
    rhs.set({1, 0}, 7.0);
    rhs.set({1, 1}, 8.0);
    lhs.set_requires_grad(true);
    rhs.set_requires_grad(true);
    Tensor prod = synapse::matmul(lhs, rhs);
    synapse::sum(prod).backward();
    assert_near(lhs.grad()->at({0, 0}), 11.0, 1e-9, "matmul grad lhs");
    assert_near(lhs.grad()->at({1, 1}), 15.0, 1e-9, "matmul grad lhs 2");
    assert_near(rhs.grad()->at({0, 0}), 4.0, 1e-9, "matmul grad rhs");
    assert_near(rhs.grad()->at({1, 1}), 6.0, 1e-9, "matmul grad rhs 2");

    Tensor h({2});
    h.set({0}, 3.0);
    h.set({1}, 4.0);
    h.set_requires_grad(true);
    h.register_hook([](Tensor& grad) { grad = grad * 0.5; });
    synapse::sum(h).backward();
    assert_near(h.grad()->at({0}), 0.5, 1e-9, "hook grad");

    Tensor n({2});
    n.set({0}, 1.0);
    n.set({1}, 2.0);
    n.set_requires_grad(true);
    {
        Tensor::NoGradGuard guard;
        Tensor out = n * 3.0;
        assert_true(!out.requires_grad(), "no_grad disables graph");
    }
    Tensor detached = n.detach();
    Tensor out = detached * 2.0;
    assert_true(!out.requires_grad(), "detach disables grad");

    Tensor acc({2});
    acc.set({0}, 1.5);
    acc.set({1}, -2.0);
    acc.set_requires_grad(true);
    Tensor acc_out = synapse::sum(acc * acc + acc * 2.0);
    acc_out.backward(true);
    double grad_first = acc.grad()->at({0});
    acc.zero_grad();
    acc_out.backward();
    assert_near(acc.grad()->at({0}), grad_first, 1e-9, "retain_graph keeps backward");

    Tensor gc = make_sequential_values_tensor({2, 2}, 0.5);
    bool check_ok = synapse::grad_check([](const Tensor& t) { return synapse::sum(t * t); }, gc);
    assert_true(check_ok, "grad_check");

    Tensor graph_x({2});
    graph_x.set({0}, 1.0);
    graph_x.set({1}, 2.0);
    graph_x.set_requires_grad(true);
    Tensor graph_out = synapse::sum(graph_x * 3.0);
    std::string dot = graph_out.grad_graph_dot();
    assert_true(dot.find("digraph") != std::string::npos, "graph dot");

    std::cout << "All autograd tests passed.\n";
    return 0;
}
