#include "synapse/tensor.hpp"

#include "synapse/simd.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace synapse {

namespace {

size_t compute_numel(const Tensor::Shape& shape) {
    if (shape.empty()) {
        return 1;
    }
    return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<>());
}

Tensor::Strides contiguous_strides(const Tensor::Shape& shape) {
    Tensor::Strides strides(shape.size(), 0);
    if (shape.empty()) {
        return strides;
    }
    size_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

void increment_index(std::vector<size_t>& index, const Tensor::Shape& shape) {
    if (shape.empty()) {
        return;
    }
    for (size_t i = shape.size(); i-- > 0;) {
        index[i] += 1;
        if (index[i] < shape[i]) {
            return;
        }
        index[i] = 0;
    }
}

Tensor::Shape broadcast_shape(const Tensor::Shape& lhs, const Tensor::Shape& rhs) {
    size_t ndim = std::max(lhs.size(), rhs.size());
    Tensor::Shape out(ndim, 1);
    for (size_t i = 0; i < ndim; ++i) {
        size_t lhs_dim = (i < ndim - lhs.size()) ? 1 : lhs[i - (ndim - lhs.size())];
        size_t rhs_dim = (i < ndim - rhs.size()) ? 1 : rhs[i - (ndim - rhs.size())];
        if (lhs_dim == rhs_dim || lhs_dim == 1 || rhs_dim == 1) {
            out[i] = std::max(lhs_dim, rhs_dim);
        } else {
            throw std::invalid_argument("Shapes are not broadcastable");
        }
    }
    return out;
}

std::shared_ptr<MemoryPool> g_memory_pool;

enum class BinaryOp {
    Add,
    Sub,
    Mul,
    Div
};

double apply_binary(BinaryOp op, double lhs, double rhs) {
    switch (op) {
        case BinaryOp::Add:
            return lhs + rhs;
        case BinaryOp::Sub:
            return lhs - rhs;
        case BinaryOp::Mul:
            return lhs * rhs;
        case BinaryOp::Div:
            return lhs / rhs;
        default:
            throw std::invalid_argument("Unknown binary op");
    }
}

size_t offset_for_index(const Tensor& tensor, const std::vector<size_t>& index) {
    size_t offset = 0;
    const auto& strides = tensor.strides();
    for (size_t i = 0; i < strides.size(); ++i) {
        offset += index[i] * strides[i];
    }
    return offset;
}

double read_value(const Tensor& tensor, size_t offset) {
    switch (tensor.dtype()) {
        case DType::Float32:
            return static_cast<double>(tensor.data<float>()[offset]);
        case DType::Float64:
            return tensor.data<double>()[offset];
        case DType::Int32:
            return static_cast<double>(tensor.data<int32_t>()[offset]);
        case DType::Int64:
            return static_cast<double>(tensor.data<int64_t>()[offset]);
        default:
            throw std::invalid_argument("Unsupported dtype");
    }
}

void write_value(Tensor& tensor, size_t offset, double value) {
    switch (tensor.dtype()) {
        case DType::Float32:
            tensor.data<float>()[offset] = static_cast<float>(value);
            break;
        case DType::Float64:
            tensor.data<double>()[offset] = value;
            break;
        case DType::Int32:
            tensor.data<int32_t>()[offset] = static_cast<int32_t>(value);
            break;
        case DType::Int64:
            tensor.data<int64_t>()[offset] = static_cast<int64_t>(value);
            break;
        default:
            throw std::invalid_argument("Unsupported dtype");
    }
}

Tensor elementwise_binary(const Tensor& lhs, const Tensor& rhs, BinaryOp op) {
    if (lhs.dtype() != rhs.dtype()) {
        throw std::invalid_argument("DType mismatch for elementwise operation");
    }

    Tensor::Shape out_shape = broadcast_shape(lhs.shape(), rhs.shape());
    Tensor lhs_view = lhs.broadcast_to(out_shape);
    Tensor rhs_view = rhs.broadcast_to(out_shape);
    Tensor out(out_shape, lhs.dtype(), Tensor::global_memory_pool());

    if (lhs_view.is_contiguous() && rhs_view.is_contiguous() && out.is_contiguous()) {
        size_t count = out.numel();
        if (out.dtype() == DType::Float32) {
            switch (op) {
                case BinaryOp::Add:
                    simd::add(lhs_view.data<float>(), rhs_view.data<float>(), out.data<float>(), count);
                    return out;
                case BinaryOp::Sub:
                    simd::sub(lhs_view.data<float>(), rhs_view.data<float>(), out.data<float>(), count);
                    return out;
                case BinaryOp::Mul:
                    simd::mul(lhs_view.data<float>(), rhs_view.data<float>(), out.data<float>(), count);
                    return out;
                case BinaryOp::Div:
                    simd::div(lhs_view.data<float>(), rhs_view.data<float>(), out.data<float>(), count);
                    return out;
                default:
                    break;
            }
        }
        if (out.dtype() == DType::Float64) {
            switch (op) {
                case BinaryOp::Add:
                    simd::add(lhs_view.data<double>(), rhs_view.data<double>(), out.data<double>(), count);
                    return out;
                case BinaryOp::Sub:
                    simd::sub(lhs_view.data<double>(), rhs_view.data<double>(), out.data<double>(), count);
                    return out;
                case BinaryOp::Mul:
                    simd::mul(lhs_view.data<double>(), rhs_view.data<double>(), out.data<double>(), count);
                    return out;
                case BinaryOp::Div:
                    simd::div(lhs_view.data<double>(), rhs_view.data<double>(), out.data<double>(), count);
                    return out;
                default:
                    break;
            }
        }
    }

    std::vector<size_t> index(out.shape().size(), 0);
    for (size_t i = 0; i < out.numel(); ++i) {
        size_t lhs_offset = offset_for_index(lhs_view, index);
        size_t rhs_offset = offset_for_index(rhs_view, index);
        size_t out_offset = offset_for_index(out, index);
        double value = apply_binary(op, read_value(lhs_view, lhs_offset), read_value(rhs_view, rhs_offset));
        write_value(out, out_offset, value);
        increment_index(index, out.shape());
    }

    return out;
}

Tensor elementwise_scalar(const Tensor& lhs, double scalar, BinaryOp op) {
    Tensor out(lhs.shape(), lhs.dtype(), Tensor::global_memory_pool());

    if (lhs.is_contiguous() && out.is_contiguous()) {
        size_t count = out.numel();
        if (out.dtype() == DType::Float32) {
            switch (op) {
                case BinaryOp::Add:
                    simd::add_scalar(lhs.data<float>(), static_cast<float>(scalar), out.data<float>(), count);
                    return out;
                case BinaryOp::Mul:
                    simd::mul_scalar(lhs.data<float>(), static_cast<float>(scalar), out.data<float>(), count);
                    return out;
                default:
                    break;
            }
            for (size_t i = 0; i < count; ++i) {
                out.data<float>()[i] = static_cast<float>(apply_binary(op, lhs.data<float>()[i], scalar));
            }
            return out;
        }
        if (out.dtype() == DType::Float64) {
            switch (op) {
                case BinaryOp::Add:
                    simd::add_scalar(lhs.data<double>(), scalar, out.data<double>(), count);
                    return out;
                case BinaryOp::Mul:
                    simd::mul_scalar(lhs.data<double>(), scalar, out.data<double>(), count);
                    return out;
                default:
                    break;
            }
            for (size_t i = 0; i < count; ++i) {
                out.data<double>()[i] = apply_binary(op, lhs.data<double>()[i], scalar);
            }
            return out;
        }
    }

    std::vector<size_t> index(lhs.shape().size(), 0);
    for (size_t i = 0; i < out.numel(); ++i) {
        size_t offset = offset_for_index(lhs, index);
        double value = apply_binary(op, read_value(lhs, offset), scalar);
        write_value(out, offset_for_index(out, index), value);
        increment_index(index, lhs.shape());
    }

    return out;
}

Tensor reduce_op(const Tensor& input, std::optional<size_t> axis, bool keepdims, const std::function<double(double, double)>& fn, double init) {
    if (!axis.has_value()) {
        Tensor out({}, input.dtype(), Tensor::global_memory_pool());
        double acc = init;
        std::vector<size_t> index(input.shape().size(), 0);
        for (size_t i = 0; i < input.numel(); ++i) {
            size_t offset = offset_for_index(input, index);
            acc = fn(acc, read_value(input, offset));
            increment_index(index, input.shape());
        }
        write_value(out, 0, acc);
        return out;
    }

    if (input.shape().empty()) {
        throw std::invalid_argument("Cannot reduce scalar with axis");
    }

    size_t axis_index = axis.value();
    if (axis_index >= input.shape().size()) {
        throw std::invalid_argument("Axis out of range");
    }

    Tensor::Shape out_shape = input.shape();
    if (keepdims) {
        out_shape[axis_index] = 1;
    } else {
        out_shape.erase(out_shape.begin() + static_cast<long>(axis_index));
    }

    Tensor out(out_shape, input.dtype(), Tensor::global_memory_pool());

    std::vector<size_t> out_index(out_shape.size(), 0);
    size_t out_numel = out.numel();
    for (size_t i = 0; i < out_numel; ++i) {
        double acc = init;
        for (size_t k = 0; k < input.shape()[axis_index]; ++k) {
            std::vector<size_t> in_index;
            in_index.reserve(input.shape().size());
            for (size_t d = 0; d < input.shape().size(); ++d) {
                if (d == axis_index) {
                    in_index.push_back(k);
                } else {
                    size_t out_pos = d;
                    if (!keepdims && d > axis_index) {
                        out_pos -= 1;
                    }
                    in_index.push_back(out_index[out_pos]);
                }
            }
            acc = fn(acc, read_value(input, offset_for_index(input, in_index)));
        }
        write_value(out, offset_for_index(out, out_index), acc);
        increment_index(out_index, out_shape);
    }

    return out;
}

} // namespace

size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Float32:
            return sizeof(float);
        case DType::Float64:
            return sizeof(double);
        case DType::Int32:
            return sizeof(int32_t);
        case DType::Int64:
            return sizeof(int64_t);
        default:
            throw std::invalid_argument("Unknown dtype");
    }
}

std::string dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Float32:
            return "float32";
        case DType::Float64:
            return "float64";
        case DType::Int32:
            return "int32";
        case DType::Int64:
            return "int64";
        default:
            return "unknown";
    }
}

Slice Slice::all(size_t dim_size) {
    return {0, dim_size, 1};
}

Tensor::Tensor(const Shape& shape, DType dtype)
    : Tensor(shape, dtype, global_memory_pool()) {}

Tensor::Tensor(const Shape& shape, DType dtype, std::shared_ptr<MemoryPool> pool)
    : shape_(shape), strides_(contiguous_strides(shape)), dtype_(dtype), pool_(std::move(pool)) {
    size_t bytes = compute_numel(shape_) * dtype_size(dtype_);
    if (bytes > 0) {
        if (!pool_) {
            data_.reset(new uint8_t[bytes], std::default_delete<uint8_t[]>());
        } else {
            data_ = pool_->allocate(bytes);
        }
    }
}

Tensor::Tensor(Shape shape, Strides strides, DType dtype, size_t offset, std::shared_ptr<uint8_t> data, std::shared_ptr<MemoryPool> pool)
    : shape_(std::move(shape)),
      strides_(std::move(strides)),
      dtype_(dtype),
      offset_(offset),
      data_(std::move(data)),
      pool_(std::move(pool)) {}

size_t Tensor::ndim() const {
    return shape_.size();
}

const Tensor::Shape& Tensor::shape() const {
    return shape_;
}

const Tensor::Strides& Tensor::strides() const {
    return strides_;
}

DType Tensor::dtype() const {
    return dtype_;
}

size_t Tensor::numel() const {
    return compute_numel(shape_);
}

bool Tensor::is_contiguous() const {
    return strides_ == contiguous_strides(shape_);
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        return view();
    }
    return clone();
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    if (compute_numel(new_shape) != numel()) {
        throw std::invalid_argument("Reshape size mismatch");
    }
    if (!is_contiguous()) {
        return clone().reshape(new_shape);
    }
    return Tensor(new_shape, contiguous_strides(new_shape), dtype_, offset_, data_, pool_);
}

Tensor Tensor::transpose(const std::vector<size_t>& axes) const {
    if (shape_.empty()) {
        return view();
    }

    std::vector<size_t> perm = axes;
    if (perm.empty()) {
        perm.resize(shape_.size());
        std::iota(perm.begin(), perm.end(), 0);
        std::reverse(perm.begin(), perm.end());
    }

    if (perm.size() != shape_.size()) {
        throw std::invalid_argument("Transpose axes size mismatch");
    }

    Shape new_shape(shape_.size());
    Strides new_strides(shape_.size());
    std::vector<bool> seen(shape_.size(), false);
    for (size_t i = 0; i < perm.size(); ++i) {
        if (perm[i] >= shape_.size() || seen[perm[i]]) {
            throw std::invalid_argument("Invalid transpose axes");
        }
        seen[perm[i]] = true;
        new_shape[i] = shape_[perm[i]];
        new_strides[i] = strides_[perm[i]];
    }

    return Tensor(new_shape, new_strides, dtype_, offset_, data_, pool_);
}

Tensor Tensor::slice(const std::vector<Slice>& slices) const {
    if (slices.size() != shape_.size()) {
        throw std::invalid_argument("Slice rank mismatch");
    }

    Shape new_shape;
    Strides new_strides;
    new_shape.reserve(shape_.size());
    new_strides.reserve(shape_.size());

    size_t new_offset = offset_;
    for (size_t i = 0; i < slices.size(); ++i) {
        const auto& slice = slices[i];
        if (slice.step == 0 || slice.start >= slice.end || slice.end > shape_[i]) {
            throw std::invalid_argument("Invalid slice");
        }
        size_t length = (slice.end - slice.start + slice.step - 1) / slice.step;
        new_shape.push_back(length);
        new_strides.push_back(strides_[i] * slice.step);
        new_offset += slice.start * strides_[i];
    }

    return Tensor(new_shape, new_strides, dtype_, new_offset, data_, pool_);
}

Tensor Tensor::broadcast_to(const Shape& new_shape) const {
    if (new_shape.size() < shape_.size()) {
        throw std::invalid_argument("Cannot broadcast to smaller rank");
    }

    size_t ndim = new_shape.size();
    size_t offset = ndim - shape_.size();
    Shape out_shape = new_shape;
    Strides out_strides(ndim, 0);

    for (size_t i = 0; i < ndim; ++i) {
        size_t dim = (i < offset) ? 1 : shape_[i - offset];
        size_t stride = (i < offset) ? 0 : strides_[i - offset];
        if (dim == out_shape[i]) {
            out_strides[i] = stride;
        } else if (dim == 1) {
            out_strides[i] = 0;
        } else {
            throw std::invalid_argument("Broadcast dimension mismatch");
        }
    }

    return Tensor(out_shape, out_strides, dtype_, offset_, data_, pool_);
}

Tensor Tensor::clone() const {
    Tensor out(shape_, dtype_, pool_ ? pool_ : global_memory_pool());

    if (numel() == 0) {
        return out;
    }

    if (is_contiguous()) {
        std::memcpy(out.raw_data(), raw_data(), numel() * dtype_size(dtype_));
        return out;
    }

    std::vector<size_t> index(shape_.size(), 0);
    for (size_t i = 0; i < numel(); ++i) {
        size_t offset = offset_for_index(*this, index);
        write_value(out, offset_for_index(out, index), read_value(*this, offset));
        increment_index(index, shape_);
    }

    return out;
}

Tensor Tensor::view() const {
    return Tensor(shape_, strides_, dtype_, offset_, data_, pool_);
}

double Tensor::at(const std::vector<size_t>& indices) const {
    size_t offset = compute_offset(indices);
    return read_value(*this, offset);
}

void Tensor::set(const std::vector<size_t>& indices, double value) {
    size_t offset = compute_offset(indices);
    write_value(*this, offset, value);
}

std::vector<uint8_t> Tensor::serialize() const {
    Tensor contig = is_contiguous() ? view() : clone();
    uint64_t ndim = static_cast<uint64_t>(contig.shape_.size());
    uint64_t data_bytes = static_cast<uint64_t>(contig.numel() * dtype_size(contig.dtype_));

    std::vector<uint8_t> bytes;
    bytes.reserve(6 + ndim * sizeof(uint64_t) + static_cast<size_t>(data_bytes));
    bytes.insert(bytes.end(), {'S', 'Y', 'N', '0'});
    bytes.push_back(1);
    bytes.push_back(static_cast<uint8_t>(contig.dtype_));

    auto append_u64 = [&bytes](uint64_t value) {
        for (size_t i = 0; i < sizeof(uint64_t); ++i) {
            bytes.push_back(static_cast<uint8_t>((value >> (8 * i)) & 0xFF));
        }
    };

    append_u64(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        append_u64(static_cast<uint64_t>(contig.shape_[i]));
    }

    const uint8_t* data_ptr = contig.raw_data();
    bytes.insert(bytes.end(), data_ptr, data_ptr + static_cast<size_t>(data_bytes));
    return bytes;
}

Tensor Tensor::deserialize(const std::vector<uint8_t>& bytes) {
    if (bytes.size() < 6) {
        throw std::invalid_argument("Serialized tensor data too small");
    }
    if (!(bytes[0] == 'S' && bytes[1] == 'Y' && bytes[2] == 'N' && bytes[3] == '0')) {
        throw std::invalid_argument("Invalid tensor header");
    }
    if (bytes[4] != 1) {
        throw std::invalid_argument("Unsupported tensor version");
    }
    DType dtype = static_cast<DType>(bytes[5]);

    size_t offset = 6;
    auto read_u64 = [&bytes, &offset]() {
        if (offset + sizeof(uint64_t) > bytes.size()) {
            throw std::invalid_argument("Serialized tensor truncated");
        }
        uint64_t value = 0;
        for (size_t i = 0; i < sizeof(uint64_t); ++i) {
            value |= static_cast<uint64_t>(bytes[offset + i]) << (8 * i);
        }
        offset += sizeof(uint64_t);
        return value;
    };

    uint64_t ndim = read_u64();
    Shape shape;
    shape.reserve(ndim);
    for (size_t i = 0; i < ndim; ++i) {
        shape.push_back(static_cast<size_t>(read_u64()));
    }

    Tensor out(shape, dtype, global_memory_pool());
    size_t data_bytes = out.numel() * dtype_size(dtype);
    if (offset + data_bytes > bytes.size()) {
        throw std::invalid_argument("Serialized tensor data truncated");
    }
    if (data_bytes > 0) {
        std::memcpy(out.raw_data(), bytes.data() + offset, data_bytes);
    }
    return out;
}

void Tensor::save(const std::string& path) const {
    std::vector<uint8_t> bytes = serialize();
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file for tensor save");
    }
    out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
}

Tensor Tensor::load(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("Failed to open file for tensor load");
    }
    std::streamsize size = in.tellg();
    in.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (size > 0 && !in.read(reinterpret_cast<char*>(bytes.data()), size)) {
        throw std::runtime_error("Failed to read tensor file");
    }
    return deserialize(bytes);
}

void Tensor::set_global_memory_pool(std::shared_ptr<MemoryPool> pool) {
    g_memory_pool = std::move(pool);
}

std::shared_ptr<MemoryPool> Tensor::global_memory_pool() {
    return g_memory_pool;
}

uint8_t* Tensor::raw_data() const {
    if (!data_) {
        return nullptr;
    }
    return data_.get() + offset_ * dtype_size(dtype_);
}

size_t Tensor::compute_offset(const std::vector<size_t>& indices) const {
    if (indices.size() != shape_.size()) {
        throw std::invalid_argument("Index rank mismatch");
    }
    size_t offset = 0;
    for (size_t i = 0; i < shape_.size(); ++i) {
        if (indices[i] >= shape_[i]) {
            throw std::out_of_range("Index out of bounds");
        }
        offset += indices[i] * strides_[i];
    }
    return offset;
}

Tensor add(const Tensor& lhs, const Tensor& rhs) {
    return elementwise_binary(lhs, rhs, BinaryOp::Add);
}

Tensor sub(const Tensor& lhs, const Tensor& rhs) {
    return elementwise_binary(lhs, rhs, BinaryOp::Sub);
}

Tensor mul(const Tensor& lhs, const Tensor& rhs) {
    return elementwise_binary(lhs, rhs, BinaryOp::Mul);
}

Tensor div(const Tensor& lhs, const Tensor& rhs) {
    return elementwise_binary(lhs, rhs, BinaryOp::Div);
}

Tensor add(const Tensor& lhs, double scalar) {
    return elementwise_scalar(lhs, scalar, BinaryOp::Add);
}

Tensor sub(const Tensor& lhs, double scalar) {
    return elementwise_scalar(lhs, scalar, BinaryOp::Sub);
}

Tensor mul(const Tensor& lhs, double scalar) {
    return elementwise_scalar(lhs, scalar, BinaryOp::Mul);
}

Tensor div(const Tensor& lhs, double scalar) {
    return elementwise_scalar(lhs, scalar, BinaryOp::Div);
}

Tensor matmul(const Tensor& lhs, const Tensor& rhs) {
    if (lhs.ndim() != 2 || rhs.ndim() != 2) {
        throw std::invalid_argument("matmul expects 2D tensors");
    }
    if (lhs.shape()[1] != rhs.shape()[0]) {
        throw std::invalid_argument("matmul dimension mismatch");
    }
    if (lhs.dtype() != rhs.dtype()) {
        throw std::invalid_argument("matmul dtype mismatch");
    }

    size_t m = lhs.shape()[0];
    size_t k_dim = lhs.shape()[1];
    size_t n = rhs.shape()[1];

    Tensor out({m, n}, lhs.dtype(), Tensor::global_memory_pool());

    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (size_t k = 0; k < k_dim; ++k) {
                double a_val = read_value(lhs, i * lhs.strides()[0] + k * lhs.strides()[1]);
                double b_val = read_value(rhs, k * rhs.strides()[0] + j * rhs.strides()[1]);
                acc += a_val * b_val;
            }
            write_value(out, i * out.strides()[0] + j * out.strides()[1], acc);
        }
    }

    return out;
}

Tensor sum(const Tensor& input, std::optional<size_t> axis, bool keepdims) {
    return reduce_op(input, axis, keepdims, [](double a, double b) { return a + b; }, 0.0);
}

Tensor mean(const Tensor& input, std::optional<size_t> axis, bool keepdims) {
    Tensor total = sum(input, axis, keepdims);
    double denom = axis.has_value() ? static_cast<double>(input.shape()[axis.value()]) : static_cast<double>(input.numel());
    return div(total, denom);
}

Tensor max(const Tensor& input, std::optional<size_t> axis, bool keepdims) {
    double init = -std::numeric_limits<double>::infinity();
    return reduce_op(input, axis, keepdims, [](double a, double b) { return std::max(a, b); }, init);
}

Tensor apply(const Tensor& input, const std::function<double(double)>& fn) {
    Tensor out(input.shape(), input.dtype(), Tensor::global_memory_pool());

    std::vector<size_t> index(input.shape().size(), 0);
    for (size_t i = 0; i < input.numel(); ++i) {
        size_t offset = offset_for_index(input, index);
        write_value(out, offset_for_index(out, index), fn(read_value(input, offset)));
        increment_index(index, input.shape());
    }

    return out;
}

Tensor operator+(const Tensor& lhs, const Tensor& rhs) {
    return add(lhs, rhs);
}

Tensor operator-(const Tensor& lhs, const Tensor& rhs) {
    return sub(lhs, rhs);
}

Tensor operator*(const Tensor& lhs, const Tensor& rhs) {
    return mul(lhs, rhs);
}

Tensor operator/(const Tensor& lhs, const Tensor& rhs) {
    return div(lhs, rhs);
}

Tensor operator+(const Tensor& lhs, double scalar) {
    return add(lhs, scalar);
}

Tensor operator-(const Tensor& lhs, double scalar) {
    return sub(lhs, scalar);
}

Tensor operator*(const Tensor& lhs, double scalar) {
    return mul(lhs, scalar);
}

Tensor operator/(const Tensor& lhs, double scalar) {
    return div(lhs, scalar);
}

} // namespace synapse
