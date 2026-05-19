#include "synapse/profiler.hpp"

namespace synapse {

Profiler& Profiler::instance() {
    static Profiler profiler;
    return profiler;
}

void Profiler::record(const std::string& name, std::chrono::nanoseconds duration) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto& entry = entries_[name];
    entry.count += 1;
    entry.total += duration;
    if (duration > entry.max) {
        entry.max = duration;
    }
}

std::vector<ProfileStats> Profiler::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<ProfileStats> stats;
    stats.reserve(entries_.size());
    for (const auto& [name, entry] : entries_) {
        stats.push_back({name, entry.count, entry.total, entry.max});
    }
    return stats;
}

void Profiler::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    entries_.clear();
}

ProfileScope::ProfileScope(std::string name)
    : name_(std::move(name)), start_(std::chrono::steady_clock::now()) {}

ProfileScope::~ProfileScope() {
    auto end = std::chrono::steady_clock::now();
    Profiler::instance().record(name_, std::chrono::duration_cast<std::chrono::nanoseconds>(end - start_));
}

} // namespace synapse
