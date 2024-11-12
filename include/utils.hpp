//
// Created by JCW on 8/10/2024.
//

#ifndef RT_PROJECT_UTILS_HPP
#define RT_PROJECT_UTILS_HPP
#include <chrono>
#include <thread>
#include <source_location>
#include <concepts>
#include <iostream>
#include <functional>
#include <any>
#include <fstream>

// NVCC is not happy about it
//#include <format>
#include <map>
#include <ranges>
#include <optional>
namespace utils {
    /// helper template to construct move-only class
    /// Usage: class Derived : private NonCopyable<Derived> {...}, or consult C++ CRTP
    template <class T>
    class NonCopyable {
    public:
        NonCopyable (const NonCopyable &) = delete;
        T & operator = (const T &) = delete;

    protected:
        NonCopyable () = default;
        ~NonCopyable () = default; /// Protected non-virtual destructor
    };
    enum class LogLevel{
        eVerbose,
        eLog,
        eWarn,
        eErr
    };
    /// simple function to log something with adjustable level of severity.
    /// Usage: log_and_pause(content, pause_time_ms) or log_and_pause<LogLevel::...>(content, pause_time_ms)
    template <LogLevel lvl=LogLevel::eLog>
    void log_and_pause(const std::optional<std::string_view> &prompt = std::nullopt,
                       size_t sleepMs = 1000,
                       const std::source_location& location = std::source_location::current()){
        bool is_cout = true;
        if constexpr (lvl<=LogLevel::eLog) {
            std::cout << "Log:";
        } else {
            is_cout = false;
            if constexpr (lvl==LogLevel::eErr) {
                std::cerr << "Error:";
            } else {
                std::cerr << "Unknown:";
            }
        }
        if (is_cout) {
            if constexpr (lvl>LogLevel::eVerbose) {
                std::cout << location.file_name() << ':'
                          << location.line() << ' ';
            }
            std::cout << (prompt.has_value() ? prompt.value() : "") << std::endl;
        } else {
            std::cerr << location.file_name() << ':'
                      << location.line() << ' '
                      << (prompt.has_value() ? prompt.value() : "") << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(sleepMs));
    }
    template <LogLevel lvl=LogLevel::eLog>
    void log(const std::optional<std::string_view> &prompt = std::nullopt,
             const std::source_location& location = std::source_location::current()){
        log_and_pause<lvl>(prompt, 0, location);
    }
#ifdef __CUDACC__

        /// thread id inside a thread block
template <int blockDimCount = 1>
__device__ unsigned int getBTId() {
    static_assert(1<=blockDimCount && blockDimCount<=3, "blockDimCount must be in [1,3]");
    auto tId = threadIdx.x;
    if constexpr (blockDimCount>1) {
        tId += threadIdx.y * blockDim.x;
    }
    if constexpr (blockDimCount>2) {
        tId += threadIdx.z * blockDim.x * blockDim.y;
    }
    return tId;
}
    /// block id inside a grid
template <int gridDimCount = 1>
__device__ unsigned int getBId() {
    static_assert(1<=gridDimCount && gridDimCount<=3, "gridDimCount must be in [1,3]");
    auto bId = blockIdx.x;
    if constexpr (gridDimCount>1) {
        bId += blockIdx.y * gridDim.x;
    }
    if constexpr (gridDimCount>2) {
        bId += blockIdx.z * gridDim.x * gridDim.y;
    }
    return bId;
}
    /// number of threads in a block
    template <int blockDimCount = 1>
    __device__ unsigned int getBlock() {
    static_assert(1<=blockDimCount && blockDimCount<=3, "blockDimCount must be in [1,3]");
    auto tBlockSize = blockDim.x;
    if constexpr (blockDimCount>1) {
        tBlockSize *= blockDim.y;
    }
    if constexpr (blockDimCount>2) {
        tBlockSize *= blockDim.z;
    }
    return tBlockSize;
}
    /// thread id inside a grid
template <int blockDimCount = 1, int gridDimCount = 1>
__device__ unsigned int getTId() {
    return getBTId<blockDimCount>() + getBId<gridDimCount>()*getBlock<blockDimCount>();
}

    /// number of blocks in a grid
    template <int gridDimCount = 1>
__device__ unsigned int getBGrid() {
    static_assert(1<=gridDimCount && gridDimCount<=3, "gridDimCount must be in [1,3]");
    auto gSize = gridDim.x;
    if constexpr (gridDimCount>1) {
        gSize *= gridDim.y;
    }
    if constexpr (gridDimCount>2) {
        gSize *= gridDim.z;
    }
    return gSize;
}
    /// number of threads in a grid
    template <int blockDimCount = 1, int gridDimCount = 1>
__device__ unsigned int getGrid() {
    return getBlock<blockDimCount>() * getBGrid<gridDimCount>();
}

    /// helper function to check results of CUDA API calls, pause 1 sec and emit log if not cudaSuccess
    /// Usage: cu_ensure(cudaAPICall(...))
    inline void cu_ensure(
            const cudaError_t result,
            const std::optional<std::string_view> &prompt = std::nullopt,
            const std::source_location& location = std::source_location::current()) {
        if (result != cudaSuccess) [[unlikely]]{
            std::string prompt_str = prompt.has_value() ?
                    //std::format("{} CUDA API error: {}", prompt.value(), cudaGetErrorString(result)) :
                    //std::format("CUDA API error: {}", cudaGetErrorString(result));
                    std::string{prompt.value()} + "CUDA API error: " + cudaGetErrorString(result) :
                    std::string{"CUDA API error: "} + cudaGetErrorString(result);
            cudaDeviceReset();
            log_and_pause<LogLevel::eErr>(prompt_str, 1000, location);
            std::abort();
        }
    }
    /// helper function to check last CUDA API error
    /// Usage: cu_check()
    inline void cu_check(const std::source_location& location = std::source_location::current()) {
        cu_ensure(cudaGetLastError(), "cu_check: ", location);
    }
    template <typename T>
    class CuArrayRAII: private NonCopyable<CuArrayRAII<T>> {
    public:
        explicit CuArrayRAII(const T* src, const size_t n=1):n(n) {
            cu_ensure(cudaMalloc(&cudaPtr, sizeof(T) * n));
            if (src != nullptr) {
                cu_ensure(cudaMemcpy(cudaPtr, src, sizeof(T) * n, cudaMemcpyHostToDevice));
            }
        }
        ~CuArrayRAII() {
            cudaFree(cudaPtr);
        }
        T* cudaPtr{};
        size_t n;
    };

    template <class T>
    class NaiveVector {
    public:
        __host__ __device__ NaiveVector(): NaiveVector(1) {}
        __host__ __device__ explicit NaiveVector(const int capacity): capacity_(capacity) {
            data = new T[capacity_];
        }

        __host__ __device__ ~NaiveVector() {
            delete[] data;
        }

        __host__ __device__ NaiveVector(const NaiveVector & other) {
            *this = other;
        }
        __host__ __device__ NaiveVector & operator = (const NaiveVector &other) {
            if (this != &other) {
                capacity_ = other.capacity_;
                count = other.count;
                delete[] data;
                data = new T[capacity_];
                memcpy(data, other.data, sizeof(T) * count);
            }
            return *this;
        }
        __host__ __device__ void push(const T &elem) {
            if (count == capacity_) {
                auto new_data = new T[capacity_+10];
                memcpy(new_data, data, sizeof(T) * count);
                delete[] data;
                data = new_data;
            }
            data[count] = elem;
            count += 1;
        }
        __host__ __device__ T pop() {
            if (count > 0) {
                count -= 1;
                return data[count];
            }
            return {};
        }
        __host__ __device__ T* begin() {
            return data;
        }
        __host__ __device__ T* end() {
            return data + count-1;
        }

        T* data{};
        int count{};
    private:
        int capacity_{};
    };

    template <int number>
    consteval unsigned int integer_log2() {
        if constexpr (number==1) {
            return 0;
        }
        if constexpr (number==2) {
            return 1;
        }
        if constexpr (number==4) {
            return 2;
        }
        if constexpr (number==8) {
            return 3;
        }
        if constexpr (number==16) {
            return 4;
        }
        if constexpr (number==32) {
            return 5;
        } else {
            //static_assert(false);
            return 0;
        }
    }

    template <int threads_per_work=1>
    __host__ __device__ unsigned int tId_to_workId(const unsigned int tid) {
        return tid>>integer_log2<threads_per_work>();
    }
    template <int threads_per_work=1>
    __host__ __device__ unsigned int tId_to_warp_mask(const unsigned int tid) {
        const unsigned int laneId = tid & 0x1f;
        // 0...0        1...1       0...0
        // left threads_per_work    right
        const unsigned int right_bits = tId_to_workId<threads_per_work>(laneId)*threads_per_work;
        const unsigned int left_bits = 32-(right_bits+threads_per_work);
        return (0xffffffff>>right_bits<<right_bits) & (0xffffffff<<left_bits>>left_bits);
    }
#endif

}
#endif //RT_PROJECT_UTILS_HPP
