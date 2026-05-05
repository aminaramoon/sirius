#include <cuda_runtime.h>

#include <cassert>
#include <stdexcept>
#include <utility>

namespace sirius::cuda {

class cuda_event {
 private:
  cudaEvent_t event_{nullptr};

  static void check(cudaError_t status)
  {
    if (status != cudaSuccess) { throw std::runtime_error(cudaGetErrorString(status)); }
  }

 public:
  // Constructor
  explicit cuda_event(unsigned int flags = cudaEventDefault)
  {
    check(cudaEventCreateWithFlags(&event_, flags));
  }

  // Destructor
  ~cuda_event()
  {
    if (event_ != nullptr) {
      [[maybe_unused]] cudaError_t status = cudaEventDestroy(event_);
      assert(status == cudaSuccess);
    }
  }

  // Delete copy semantics
  cuda_event(const cuda_event&)            = delete;
  cuda_event& operator=(const cuda_event&) = delete;

  // Move semantics
  cuda_event(cuda_event&& other) noexcept : event_(std::exchange(other.event_, nullptr)) {}

  cuda_event& operator=(cuda_event&& other) noexcept
  {
    if (this != &other) {
      if (event_ != nullptr) {
        [[maybe_unused]] cudaError_t status = cudaEventDestroy(event_);
        assert(status == cudaSuccess);
      }
      event_ = std::exchange(other.event_, nullptr);
    }
    return *this;
  }

  // Accessor
  [[nodiscard]] cudaEvent_t get() const noexcept { return event_; }

  // Explicit conversion — requires static_cast or direct-init contexts
  [[nodiscard]] explicit operator cudaEvent_t() const noexcept { return event_; }

  // Helpers
  void record(cudaStream_t stream = nullptr) { check(cudaEventRecord(event_, stream)); }

  void wait(cudaStream_t stream = nullptr) const { check(cudaStreamWaitEvent(stream, event_, 0)); }

  void synchronize() const { check(cudaEventSynchronize(event_)); }

  /// Returns elapsed time in milliseconds between `start` and this event.
  /// Both events must have been recorded and completed.
  [[nodiscard]] float elapsed_ms(const cuda_event& start) const
  {
    float ms = 0.f;
    check(cudaEventElapsedTime(&ms, start.get(), event_));
    return ms;
  }

  /// Query whether the event has completed (non-blocking).
  /// Returns true if complete, false if still pending.
  /// Throws on actual errors.
  [[nodiscard]] bool query() const
  {
    cudaError_t status = cudaEventQuery(event_);
    if (status == cudaSuccess) return true;
    if (status == cudaErrorNotReady) return false;
    check(status);  // throws on real errors
    return false;   // unreachable, silences warning
  }
};

}  // namespace sirius::cuda