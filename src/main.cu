#include <iostream>
#include <chrono>
#include <vector>
#include <sstream>
#include <stdexcept>

inline void cuda_check_error(cudaError_t error, const std::string filename, const std::size_t line, const std::string funcname, const std::string message = ""){
	if(error != cudaSuccess){
		std::stringstream ss;
		ss << cudaGetErrorString( error );
		if(message.length() != 0){
			ss << " : " << message;
		}
		ss << " [" << filename << ":" << line << " in " << funcname << "]";
		throw std::runtime_error(ss.str());
	}
}
#ifndef CUDA_CHECK_ERROR
#define CUDA_CHECK_ERROR(status) cuda_check_error(status, __FILE__, __LINE__, __func__)
#endif
#ifndef CUDA_CHECK_ERROR_M
#define CUDA_CHECK_ERROR_M(status, message) cuda_check_error(status, __FILE__, __LINE__, __func__, message)
#endif

template <class T>
__global__ void simple_copy_kernel(
    T* const dst_ptr,
    const T* const src_ptr,
    const std::size_t count
    ) {
  const auto tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= count) {
    return;
  }

  dst_ptr[tid] = src_ptr[tid];
}

void simple_copy(
    void* const dst_ptr,
    const void* const src_ptr,
    const std::size_t size,
    cudaStream_t cuda_stream = 0
    ) {
  const auto block_size = 1024;
  if (size % 16  == 0) {
    const auto count = size / 16;
    using data_t = ulong2;
    simple_copy_kernel<data_t><<<(count + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(reinterpret_cast<data_t*>(dst_ptr), reinterpret_cast<const data_t*>(src_ptr), count);
  } else if (size % 8 == 0) {
    const auto count = size / 8;
    using data_t = uint64_t;
    simple_copy_kernel<data_t><<<(count + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(reinterpret_cast<data_t*>(dst_ptr), reinterpret_cast<const data_t*>(src_ptr), count);
  } else if (size % 4 == 0) {
    const auto count = size / 4;
    using data_t = uint32_t;
    simple_copy_kernel<data_t><<<(count + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(reinterpret_cast<data_t*>(dst_ptr), reinterpret_cast<const data_t*>(src_ptr), count);
  } else if (size % 2 == 0) {
    const auto count = size / 2;
    using data_t = uint16_t;
    simple_copy_kernel<data_t><<<(count + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(reinterpret_cast<data_t*>(dst_ptr), reinterpret_cast<const data_t*>(src_ptr), count);
  } else {
    const auto count = size;
    using data_t = uint8_t;
    simple_copy_kernel<data_t><<<(count + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(reinterpret_cast<data_t*>(dst_ptr), reinterpret_cast<const data_t*>(src_ptr), count);
  }
}

template <class T, unsigned COPY_BLOCK_SIZE>
__global__ void simple_block_copy_kernel(
    T* const dst_ptr,
    const T* const src_ptr,
		const std::size_t count
		) {
	const std::size_t elements_per_thread_block = blockDim.x * COPY_BLOCK_SIZE;
	const std::size_t major_offset = threadIdx.x + blockIdx.x * elements_per_thread_block;
	if (major_offset >= count) {
		return;
	}
	if (major_offset + elements_per_thread_block < count) {
		T copy_block[COPY_BLOCK_SIZE];
		for (std::size_t i = 0; i < COPY_BLOCK_SIZE; i++) {
			const auto minor_offset = i * blockDim.x;
			copy_block[i] = src_ptr[major_offset + minor_offset];
		}
		for (std::size_t i = 0; i < COPY_BLOCK_SIZE; i++) {
			const auto minor_offset = i * blockDim.x;
			dst_ptr[major_offset + minor_offset] = copy_block[i];
		}
	} else {
		T copy_block[COPY_BLOCK_SIZE];
		for (std::size_t i = 0; i < COPY_BLOCK_SIZE; i++) {
			const auto minor_offset = i * blockDim.x;
			if (major_offset + minor_offset < count) {
				copy_block[i] = src_ptr[major_offset + minor_offset];
			}
		}
		for (std::size_t i = 0; i < COPY_BLOCK_SIZE; i++) {
			const auto minor_offset = i * blockDim.x;
			if (major_offset + minor_offset < count) {
				dst_ptr[major_offset + minor_offset] = copy_block[i];
			}
		}
	}
}

template <unsigned COPY_BLOCK_SIZE>
void simple_block_copy(
    void* const dst_ptr,
    const void* const src_ptr,
    const std::size_t size,
    cudaStream_t cuda_stream = 0
    ) {
  const auto block_size = 256;
	const auto grid_size = [block_size](const std::size_t count) {return ((count + block_size - 1) / block_size + COPY_BLOCK_SIZE - 1) / COPY_BLOCK_SIZE;};
  if (size % 16  == 0) {
    const auto count = size / 16;
    using data_t = ulong2;
    simple_block_copy_kernel<data_t, COPY_BLOCK_SIZE><<<grid_size(count), block_size, 0, cuda_stream>>>(reinterpret_cast<data_t*>(dst_ptr), reinterpret_cast<const data_t*>(src_ptr), count);
  } else if (size % 8 == 0) {
    const auto count = size / 8;
    using data_t = uint64_t;
    simple_block_copy_kernel<data_t, COPY_BLOCK_SIZE><<<grid_size(count), block_size, 0, cuda_stream>>>(reinterpret_cast<data_t*>(dst_ptr), reinterpret_cast<const data_t*>(src_ptr), count);
  } else if (size % 4 == 0) {
    const auto count = size / 4;
    using data_t = uint32_t;
    simple_block_copy_kernel<data_t, COPY_BLOCK_SIZE><<<grid_size(count), block_size, 0, cuda_stream>>>(reinterpret_cast<data_t*>(dst_ptr), reinterpret_cast<const data_t*>(src_ptr), count);
  } else if (size % 2 == 0) {
    const auto count = size / 2;
    using data_t = uint16_t;
    simple_block_copy_kernel<data_t, COPY_BLOCK_SIZE><<<grid_size(count), block_size, 0, cuda_stream>>>(reinterpret_cast<data_t*>(dst_ptr), reinterpret_cast<const data_t*>(src_ptr), count);
  } else {
    const auto count = size;
    using data_t = uint8_t;
    simple_block_copy_kernel<data_t, COPY_BLOCK_SIZE><<<grid_size(count), block_size, 0, cuda_stream>>>(reinterpret_cast<data_t*>(dst_ptr), reinterpret_cast<const data_t*>(src_ptr), count);
  }
}

template <class Func>
double measure_time(
    const std::size_t size,
    const Func func
    ) {
  const auto test_count = 100;
  void *dst_ptr, *src_ptr;
  cudaMalloc(&dst_ptr, size);
  cudaMalloc(&src_ptr, size);

  func(dst_ptr, src_ptr, size);

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
  const auto start_clock = std::chrono::system_clock::now();

  for (std::uint32_t i = 0; i < test_count ; i++) {
    func(dst_ptr, src_ptr, size);
  }

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
  const auto end_clock = std::chrono::system_clock::now();

  const auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock).count() * 1e-9 / test_count;

  cudaFree(dst_ptr);
  cudaFree(src_ptr);

  return elapsed_time;
}

int main() {
  std::printf("n,size_offset,size,cudaMemcpy_time,cudaMemcpy_bw,copy_kernel_time,copy_kernel_bw\n");

  for (const auto offset : std::vector<int>{0, -1, -2, 1, 2}) {
    for (std::uint32_t n = 0; n <= 30; n++) {
      const auto base_size = 1lu << n;
      if (offset < 0 && base_size <= -offset) continue;
      const auto size = base_size + offset;
      const auto cudaMemcpy_time        = measure_time(size + 1, [](void* const dst_ptr, const void* const src_ptr, const std::size_t size){cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice);});
      const auto copy_kernel_time       = measure_time(size + 1, [](void* const dst_ptr, const void* const src_ptr, const std::size_t size){simple_copy(dst_ptr, src_ptr, size);});
      const auto block_copy_kernel_time = measure_time(size + 1, [](void* const dst_ptr, const void* const src_ptr, const std::size_t size){simple_block_copy<4>(dst_ptr, src_ptr, size);});
      std::printf(
          "%u,%d,%lu,%e,%e,%e,%e,%e,%e\n",
          n,
          offset,
          size,
          cudaMemcpy_time,
          2 * size / cudaMemcpy_time,
          copy_kernel_time,
          2 * size / copy_kernel_time,
          block_copy_kernel_time,
          2 * size / block_copy_kernel_time
          );
      std::fflush(stdout);
    }
  }
}
