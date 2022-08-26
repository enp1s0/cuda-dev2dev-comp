#include <iostream>
#include <chrono>
#include <vector>

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
	const auto block_size = std::min(256lu, size);
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

template <class Func>
double measure_time(
		const std::size_t size,
		const Func func
		) {
	void *dst_ptr, *src_ptr;
	cudaMalloc(&dst_ptr, size);
	cudaMalloc(&src_ptr, size);

	func(dst_ptr, src_ptr, size);

	cudaDeviceSynchronize();
	const auto start_clock = std::chrono::system_clock::now();

	func(dst_ptr, src_ptr, size);

	cudaDeviceSynchronize();
	const auto end_clock = std::chrono::system_clock::now();

	const auto elapsed_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end_clock - start_clock).count() * 1e-9;

	cudaFree(dst_ptr);
	cudaFree(src_ptr);

	return elapsed_time;
}

int main() {
	std::printf("size,size_offset,cudaMemcpy_time,cudaMemcpy_bw,copy_kernel_time,copy_kernel_bw\n");

	for (const auto offset : std::vector<int>{0, -1, -2, 1, 2}) {
		for (std::size_t base_size = 1; base_size <= (1lu << 32); base_size <<= 1) {
			if (offset < 0 && base_size <= -offset) continue;
			const auto size = base_size + offset;
			const auto cudaMemcpy_time  = measure_time(size + 1, [](void* const dst_ptr, const void* const src_ptr, const std::size_t size){cudaMemcpy(dst_ptr, src_ptr, size, cudaMemcpyDeviceToDevice);});
			const auto copy_kernel_time = measure_time(size + 1, [](void* const dst_ptr, const void* const src_ptr, const std::size_t size){simple_copy(dst_ptr, src_ptr, size);});
			std::printf(
					"%lu,%d,%e,%e,%e,%e\n",
					size,
					offset,
					cudaMemcpy_time,
					size / cudaMemcpy_time,
					copy_kernel_time,
					size / copy_kernel_time
					);
			std::fflush(stdout);
		}
	}
}