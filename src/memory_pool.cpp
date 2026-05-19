#include "synapse/memory_pool.hpp"

#include <algorithm>
#include <new>

namespace synapse {

MemoryPool::MemoryPool(size_t block_size, size_t max_blocks)
    : block_size_(block_size), max_blocks_(max_blocks) {}

std::shared_ptr<uint8_t> MemoryPool::allocate(size_t bytes) {
    if (bytes == 0) {
        return {};
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto self = shared_from_this();
    auto make_ptr = [weak_self = std::weak_ptr<MemoryPool>(self)](uint8_t* ptr, size_t alloc_size) {
        return std::shared_ptr<uint8_t>(ptr, [weak_self, alloc_size](uint8_t* p) {
            if (auto pool = weak_self.lock()) {
                pool->release(p, alloc_size);
            } else {
                delete[] p;
            }
        });
    };

    for (auto it = free_list_.begin(); it != free_list_.end(); ++it) {
        if (it->second >= bytes) {
            uint8_t* ptr = it->first;
            size_t alloc_size = it->second;
            free_list_.erase(it);
            return make_ptr(ptr, alloc_size);
        }
    }

    size_t alloc_size = std::max(block_size_, bytes);
    if (blocks_.size() < max_blocks_) {
        blocks_.push_back({std::make_unique<uint8_t[]>(alloc_size), alloc_size});
        return make_ptr(blocks_.back().data.get(), alloc_size);
    }

    return std::shared_ptr<uint8_t>(new uint8_t[alloc_size], [alloc_size](uint8_t* p) {
        delete[] p;
    });
}

size_t MemoryPool::block_size() const {
    return block_size_;
}

size_t MemoryPool::max_blocks() const {
    return max_blocks_;
}

void MemoryPool::release(uint8_t* ptr, size_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    free_list_.push_back({ptr, bytes});
}

} // namespace synapse
