#pragma once

class Timer {
public:
    inline Timer() { clock_gettime(CLOCK_MONOTONIC, &m_begin); }
    
    inline double end()
    {
        clock_gettime(CLOCK_MONOTONIC, &m_end);
        return m_end.tv_sec - m_begin.tv_sec + (m_end.tv_nsec - m_begin.tv_nsec) / 1.0e9;
    }

private:
    struct timespec m_begin, m_end;
};
