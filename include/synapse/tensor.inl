#pragma once

#include <stdexcept>

namespace synapse {

template <typename T>
T* Tensor::data() {
    return reinterpret_cast<T*>(raw_data());
}

template <typename T>
const T* Tensor::data() const {
    return reinterpret_cast<const T*>(raw_data());
}

} // namespace synapse
