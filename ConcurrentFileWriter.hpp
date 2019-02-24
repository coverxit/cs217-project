#pragma once

class ConcurrentFileWriter final {
public:
    explicit ConcurrentFileWriter(const char* name, int size)
        : m_fd(-1)
        , m_size(size)
        , m_ptr(reinterpret_cast<uint8_t*>(MAP_FAILED))
    {
        m_fd = open(name, O_RDWR | O_CREAT | O_TRUNC, 0666);
        if (m_fd < 0) {
            perror("[ConcurrentFileWriter] open");
            return;
        }

        if (ftruncate(m_fd, size) < 0) {
            perror("[ConcurrentFileWriter] ftruncate");
            goto cleanup;
        }

        // Memory map
        m_ptr = (uint8_t*)mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, 0);
        if (m_ptr == MAP_FAILED) {
            perror("[ConcurrentFileWriter] mmap");
            goto cleanup;
        }

        return;

    cleanup:
        ::close(m_fd);
        m_fd = -1;
    }

    ~ConcurrentFileWriter() { close(); }

    operator bool() const { return m_fd >= 0; }

    void close()
    {
        if (m_ptr != MAP_FAILED) {
            munmap(m_ptr, m_size);
        }

        if (m_fd >= 0) {
            ::close(m_fd);
        }
    }

    bool write(uint8_t* buf, int size, int numOfThreads)
    {
        if (m_fd < 0) {
            std::cerr << "[ConcurrentFileWriter] File not opened." << std::endl;
            return false;
        }

        if (size < m_size) {
            std::cerr << "[ConcurrentFileWriter] Buffer size too small (" << size << " bytes)";
            std::cerr << ", expected (" << m_size << " bytes).";
            std::cerr << std::endl;
            return false;
        }

        // Too many threads?
        if (numOfThreads > size) {
            numOfThreads = size;
        }

        // Left over?
        if (size % numOfThreads) {
            ++numOfThreads;
        }

        std::vector<std::thread> threads;
        threads.reserve(numOfThreads);

        // Write file in parallel
        for (int i = 0; i < numOfThreads; ++i) {
            auto chunk = size / numOfThreads;
            auto offset = chunk * i;
            auto length = std::min(chunk, size - offset);

            threads.emplace_back([=] {
                memcpy(m_ptr + offset, buf + offset, length);
            });
        }

        for (auto& t : threads) {
            t.join();
        }

        return true;
    }

private:
    int m_fd, m_size;
    uint8_t* m_ptr;
};
