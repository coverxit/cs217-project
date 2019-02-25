#pragma once

class ConcurrentInputStream final : public ConcurrentStream {
public:
    explicit ConcurrentInputStream(const char* name)
        : ConcurrentStream(this)
    {
        m_fd = open(name, O_RDONLY);
        if (m_fd < 0) {
            perror("[ConcurrentInputStream] open");
            return;
        }

        struct stat64 buf;
        if (fstat64(m_fd, &buf) < 0) {
            perror("[ConcurrentInputStream] fstat64");
            goto cleanup;
        }

        if (!S_ISREG(buf.st_mode)) {
            std::cerr << "[ConcurrentInputStream] " << name << "is not a regluar file." << std::endl;
            goto cleanup;
        }

        if (buf.st_size > MaxFileSize) {
            std::cerr << "[ConcurrentInputStream] " << name << "'s size (" << buf.st_size << " bytes)";
            std::cerr << " is larger than supported (" << MaxFileSize << " bytes).";
            std::cerr << std::endl;
            goto cleanup;
        }

        // Memory map
        m_ptr = (uint8_t*)mmap(nullptr, buf.st_size, PROT_READ, MAP_SHARED, m_fd, 0);
        if (m_ptr == MAP_FAILED) {
            perror("[ConcurrentInputStream] mmap");
            goto cleanup;
        }

        m_size = buf.st_size;
        return;

    cleanup:
        ::close(m_fd);
        m_fd = -1;
    }

    int read(uint8_t* buf, int size, int nThreads)
    {
        return ConcurrentStream::concurrentCopy(buf, m_ptr, size, nThreads);
    }
};
