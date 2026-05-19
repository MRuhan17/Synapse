#pragma once

#include "synapse/memory_pool.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace synapse {

enum class DType : uint8_t {
    Float32 = 1,
    Float64 = 2,
    Int32 = 3,
    Int64 = 4
};

size_t dtype_size(DType dtype);
std::string dtype_name(DType dtype);

struct Slice {
    size_t start;
    size_t end;
    size_t step;

    static Slice all(size_t dim_size);
};

class Tensor {
public:
    using Shape = std::vector<size_t>;
    using Strides = std::vector<size_t>;

    Tensor() = default;
    explicit Tensor(const Shape& shape, DType dtype = DType::Float64);
    Tensor(const Shape& shape, DType dtype, std::shared_ptr<MemoryPool> pool);

    size_t ndim() const;
    const Shape& shape() const;
    const Strides& strides() const;
    DType dtype() const;
    size_t numel() const;

    bool is_contiguous() const;
    Tensor contiguous() const;
    Tensor reshape(const Shape& new_shape) const;
    Tensor transpose(const std::vector<size_t>& axes = {}) const;
    Tensor slice(const std::vector<Slice>& slices) const;
    Tensor broadcast_to(const Shape& new_shape) const;
    Tensor clone() const;
    Tensor view() const;

    double at(const std::vector<size_t>& indices) const;
    void set(const std::vector<size_t>& indices, double value);

    std::vector<uint8_t> serialize() const;
    static Tensor deserialize(const std::vector<uint8_t>& bytes);
    void save(const std::string& path) const;
    static Tensor load(const std::string& path);

    template <typename T>
    T* data();

    template <typename T>
    const T* data() const;

    static void set_global_memory_pool(std::shared_ptr<MemoryPool> pool);
    static std::shared_ptr<MemoryPool> global_memory_pool();

private:
    Shape shape_;
    Strides strides_;
    DType dtype_{DType::Float64};
    size_t offset_{0};
    std::shared_ptr<uint8_t> data_;
    std::shared_ptr<MemoryPool> pool_;

    Tensor(Shape shape, Strides strides, DType dtype, size_t offset, std::shared_ptr<uint8_t> data, std::shared_ptr<MemoryPool> pool);

    uint8_t* raw_data() const;
    size_t compute_offset(const std::vector<size_t>& indices) const;
};

Tensor add(const Tensor& lhs, const Tensor& rhs);
Tensor sub(const Tensor& lhs, const Tensor& rhs);
Tensor mul(const Tensor& lhs, const Tensor& rhs);
Tensor div(const Tensor& lhs, const Tensor& rhs);

Tensor add(const Tensor& lhs, double scalar);
Tensor sub(const Tensor& lhs, double scalar);
Tensor mul(const Tensor& lhs, double scalar);
Tensor div(const Tensor& lhs, double scalar);

Tensor matmul(const Tensor& lhs, const Tensor& rhs);

Tensor sum(const Tensor& input, std::optional<size_t> axis = std::nullopt, bool keepdims = false);
Tensor mean(const Tensor& input, std::optional<size_t> axis = std::nullopt, bool keepdims = false);
Tensor max(const Tensor& input, std::optional<size_t> axis = std::nullopt, bool keepdims = false);

Tensor apply(const Tensor& input, const std::function<double(double)>& fn);

Tensor operator+(const Tensor& lhs, const Tensor& rhs);
Tensor operator-(const Tensor& lhs, const Tensor& rhs);
Tensor operator*(const Tensor& lhs, const Tensor& rhs);
Tensor operator/(const Tensor& lhs, const Tensor& rhs);

Tensor operator+(const Tensor& lhs, double scalar);
Tensor operator-(const Tensor& lhs, double scalar);
Tensor operator*(const Tensor& lhs, double scalar);
Tensor operator/(const Tensor& lhs, double scalar);

} // namespace synapse

#include "synapse/tensor.inl"
