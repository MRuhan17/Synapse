#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <utility>
#include <vector>

namespace synapse {

class MemoryPool : public std::enable_shared_from_this<MemoryPool> {
public:
    explicit MemoryPool(size_t block_size = 1 << 20, size_t max_blocks = 64);

    std::shared_ptr<uint8_t> allocate(size_t bytes);

    size_t block_size() const;
    size_t max_blocks() const;

private:
    struct Block {
        std::unique_ptr<uint8_t[]> data;
        size_t size;
    };

    void release(uint8_t* ptr, size_t bytes);

    std::mutex mutex_;
    size_t block_size_;
    size_t max_blocks_;
    std::vector<Block> blocks_;
    std::vector<std::pair<uint8_t*, size_t>> free_list_;
};

} // namespace synapse
