#pragma once

#include <chrono>
#include <cstddef>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace synapse {

struct ProfileStats {
    std::string name;
    size_t count;
    std::chrono::nanoseconds total;
    std::chrono::nanoseconds max;
};

class Profiler {
public:
    static Profiler& instance();

    void record(const std::string& name, std::chrono::nanoseconds duration);
    std::vector<ProfileStats> snapshot() const;
    void reset();

private:
    struct Entry {
        size_t count = 0;
        std::chrono::nanoseconds total{0};
        std::chrono::nanoseconds max{0};
    };

    mutable std::mutex mutex_;
    std::unordered_map<std::string, Entry> entries_;
};

class ProfileScope {
public:
    explicit ProfileScope(std::string name);
    ~ProfileScope();

private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

} // namespace synapse
