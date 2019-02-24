#pragma once

// 32 bit signed integer, maximum 2 GB
template <uint32_t MaxFileSize = 2147483647>
class ConcurrentFileReaderT final {
public:
    explicit ConcurrentFileReaderT(const char* name)
        : m_fd(-1)
        , m_ptr((uint8_t*)MAP_FAILED)
        , m_size(0)
    {
        m_fd = open(name, O_RDONLY);
        if (m_fd < 0) {
            perror("[ConcurrentFileReader] open");
            return;
        }

        struct stat64 buf;
        if (fstat64(m_fd, &buf) < 0) {
            perror("[ConcurrentFileReader] fstat64");
            goto cleanup;
        }

        if (!S_ISREG(buf.st_mode)) {
            std::cerr << "[ConcurrentFileReader] " << name << "is not a regluar file." << std::endl;
            goto cleanup;
        }

        if (buf.st_size > MaxFileSize) {
            std::cerr << "[ConcurrentFileReader] " << name << "'s size (" << buf.st_size << " bytes)";
            std::cerr << " is larger than supported (" << MaxFileSize << " bytes).";
            std::cerr << std::endl;
            goto cleanup;
        }

        // Memory map
        m_ptr = (uint8_t*) mmap(nullptr, buf.st_size, PROT_READ, MAP_SHARED, m_fd, 0);
        if (m_ptr == MAP_FAILED) {
            perror("[ConcurrentFileReader] mmap");
            goto cleanup;
        }

        m_size = buf.st_size;
        return;

    cleanup:
        ::close(m_fd);
        m_fd = -1;
    }

    ~ConcurrentFileReaderT() { close(); }

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

    bool read(uint8_t* buf, int size, int numOfThreads)
    {
        if (m_fd < 0) {
            std::cerr << "[ConcurrentFileReader] File not opened." << std::endl;
            return false;
        }

        if (size < m_size) {
            std::cerr << "[ConcurrentFileReader] Buffer size too small (" << size << " bytes)";
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

        // Read file in parallel
        for (int i = 0; i < numOfThreads; ++i) {
            auto chunk = size / numOfThreads;
            auto offset = chunk * i;
            auto length = std::min(chunk, size - offset);

            threads.emplace_back([=] {
                memcpy(buf + offset, m_ptr + offset, length);
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

typedef ConcurrentFileReaderT<> ConcurrentFileReader;
