#include "synapse/tensor.hpp"

#include <chrono>
#include <iostream>

using synapse::Tensor;

int main() {
    const size_t dim = 512;
    Tensor a({dim, dim});
    Tensor b({dim, dim});

    double value = 1.0;
    for (size_t i = 0; i < dim; ++i) {
        for (size_t j = 0; j < dim; ++j) {
            a.set({i, j}, value);
            b.set({i, j}, value + 1.0);
            value += 1.0;
        }
    }

    auto start_add = std::chrono::steady_clock::now();
    Tensor c = a + b;
    auto end_add = std::chrono::steady_clock::now();

    auto start_mm = std::chrono::steady_clock::now();
    Tensor d = synapse::matmul(a, b);
    auto end_mm = std::chrono::steady_clock::now();

    std::chrono::duration<double> add_time = end_add - start_add;
    std::chrono::duration<double> mm_time = end_mm - start_mm;

    std::cout << "Add time (s): " << add_time.count() << "\n";
    std::cout << "Matmul time (s): " << mm_time.count() << "\n";
    std::cout << "Output checksum: " << c.at({0, 0}) + d.at({0, 0}) << "\n";
    return 0;
}
