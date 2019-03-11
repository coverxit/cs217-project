#pragma once

class ConcurrentOutputStream final : public ConcurrentStream {
public:
    explicit ConcurrentOutputStream(const char* name, int size)
        : ConcurrentStream(this, size)
        , m_written(0)
    {
        m_fd = open(name, O_RDWR | O_CREAT | O_TRUNC, 0666);
        if (m_fd < 0) {
            perror("[ConcurrentOutputStream] open");
            exit(-1);
        }

        if (ftruncate(m_fd, size) < 0) {
            perror("[ConcurrentOutputStream] ftruncate");
            goto cleanup;
        }

        // Memory map
        m_ptr = (uint8_t*)mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_SHARED, m_fd, 0);
        if (m_ptr == MAP_FAILED) {
            perror("[ConcurrentOutputStream] mmap");
            goto cleanup;
        }

        return;

    cleanup:
        ::close(m_fd);
        m_fd = -1;
        exit(-1);
    }

    int write(int offset, const uint8_t* buf, int size, int nThreads)
    {
        return ConcurrentStream::concurrentCopy(m_ptr + offset, buf, size, nThreads);
    }

    void writeNext(const uint8_t* buf, int size, int nThreads)
    {
        m_written += write(m_written, buf, size, nThreads);
    }

    int write(int initialOffset, const uint8_t* buf,
        std::vector<std::pair<int, int>>& offsets, std::vector<int>& sizes,
        int nThreads)
    {
        return ConcurrentStream::concurrentCopy(m_ptr + initialOffset, buf, offsets, sizes, nThreads);
    }

    void writeNext(const uint8_t* buf,
        std::vector<std::pair<int, int>>& offsets, std::vector<int>& sizes,
        int nThreads)
    {
        m_written += write(m_written, buf, offsets, sizes, nThreads);
    }

private:
    std::atomic_int m_written;
};
