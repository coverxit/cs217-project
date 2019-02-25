#pragma once

class ConcurrentInputStream;
class ConcurrentOutputStream;

class ConcurrentStream {
public:
    ConcurrentStream(const ConcurrentStream&) = delete;
    ConcurrentStream& operator=(const ConcurrentStream&) = delete;

protected:
    ConcurrentStream(const ConcurrentInputStream*)
        : m_reader(true)
        , m_name("ConcurrentInputStream")
        , m_fd(-1)
        , m_size(0)
        , m_ptr((uint8_t*)MAP_FAILED)
    {
    }

    ConcurrentStream(const ConcurrentOutputStream*, int size)
        : m_reader(false)
        , m_name("ConcurrentOnputStream")
        , m_fd(-1)
        , m_size(size)
        , m_ptr((uint8_t*)MAP_FAILED)
    {
    }

public:
    virtual ~ConcurrentStream() { close(); }

    operator bool() const { return m_fd >= 0; }

    int size() const { return m_size; }

    void close()
    {
        if (m_ptr != MAP_FAILED) {
            munmap(m_ptr, m_size);
        }

        if (m_fd >= 0) {
            ::close(m_fd);
        }
    }

protected:
    int concurrentCopy(uint8_t* dst, const uint8_t* src, int size, int nThreads)
    {
        if (m_fd < 0) {
            std::cerr << "[" << m_name << "] File not opened." << std::endl;
            return 0;
        }

        if (m_reader && size < m_size) {
            std::cerr << "[" << m_name << "] Buffer too small (" << size << " bytes)";
            std::cerr << ", expected (" << m_size << " bytes).";
            std::cerr << std::endl;
            return 0;
        }

        // Too many threads?
        if (nThreads > size) {
            nThreads = size;
        }

        // Left over?
        if (size % nThreads) {
            ++nThreads;
        }

        std::vector<std::thread> threads;
        threads.reserve(nThreads);

        // Read/write file in parallel
        for (int i = 0; i < nThreads; ++i) {
            auto chunk = (size - 1) / nThreads + 1;
            auto offset = chunk * i;
            auto length = std::min(chunk, size - offset);

            threads.emplace_back([=] {
                memcpy(dst + offset, src + offset, length);
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        return size;
    }

    int concurrentCopy(uint8_t* dst, const uint8_t* src,
        std::vector<std::pair<int, int>>& offsets, std::vector<int>& sizes,
        int nThreads)
    {
        std::atomic_int size(0);

        if (m_fd < 0) {
            std::cerr << "[" << m_name << "] File not opened." << std::endl;
            return 0;
        }

        if (offsets.size() != sizes.size()) {
            std::cerr << "[" << m_name << "] Number of offsets and sizes mismatch!" << std::endl;
            return 0;
        }

        if (nThreads > offsets.size()) {
            nThreads = offsets.size();
        }

        if (offsets.size() % nThreads) {
            ++nThreads;
        }

        std::vector<std::thread> threads;
        threads.reserve(nThreads);

        // Read/write file in parallel
        for (int i = 0; i < nThreads; ++i) {
            auto chunk = (offsets.size() - 1) / nThreads + 1;
            auto offset = chunk * i;
            auto length = std::min(chunk, offsets.size() - offset);

            threads.emplace_back([&offsets, &sizes, dst, src, offset, length, &size] {
                for (int j = offset; j < offset + length; ++j) {
                    memcpy(dst + offsets[j].first, src + offsets[j].second, sizes[j]);
                    size += sizes[j];
                }
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        return size;
    }

protected:
    int m_fd, m_size;
    uint8_t* m_ptr;

    bool m_reader;
    const char* m_name;
};