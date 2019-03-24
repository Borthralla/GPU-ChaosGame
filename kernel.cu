
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include "chaos_game.h"
#include "curand_kernel.h"

__global__ void setup_states(curandState * states, int seed) {
	int id = threadIdx.x + blockIdx.x * blockDim.x; 
	curand_init(seed, id, 0, &states[id]);
}

// dev_count points is repeated once per thread block.
// Each thread block should have max 32 threads.
// counts must have been allocated using cuda-allocate in global memory.
__global__ void dev_count_points(int num_points, int num_vertices, float * vertices, int num_iterations, int * counts, int length, curandState * states) {
	extern __shared__ float dev_vertices[];
	if (threadIdx.x == 0) {
		for (int i = 0; i < num_vertices * 2; i++) {
			dev_vertices[i] = vertices[i];
		}
	}
	__syncthreads();
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	curandState state = states[id];
	float radius = ((float)length) / 2;
	for (int p = 0; p < num_points; p++) {
		float current_x = radius;
		float current_y = radius;
		for (int i = 0; i < num_iterations; i++) {
			int r = (int)truncf(((float)num_vertices - 0.000001) * curand_uniform(&state));
			current_x += (dev_vertices[2 * r] - current_x) / 2;
			current_y += (dev_vertices[2 * r + 1] - current_y) / 2;
		}
		counts[length * (int)truncf(current_y) + (int)truncf(current_x)] += 1;
	}
}

int main()
{

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

	int num_blocks = 28;
	int num_threads = 128;
	int length = 4000;
	int num_vertices = 6;
	int num_points = 10000000;
	int num_iterations = 25;
	
	float * vertices = make_vertices(length, num_vertices);
	int * counts = new int[length * length]();
	float * dev_vertices;
	int * dev_counts;
	curandState * dev_states;
	cudaMalloc((void**)&dev_vertices, 2 * num_vertices * sizeof(float));
	cudaMalloc((void**)&dev_states, num_blocks * num_threads * sizeof(curandState));
	cudaMalloc((void**)&dev_counts, length * length * sizeof(int));

	cudaMemset(dev_counts, 0, length * length * sizeof(int));
	cudaMemcpy(dev_vertices, vertices, 2 * num_vertices * sizeof(float), cudaMemcpyHostToDevice);

	setup_states <<<num_blocks, num_threads >>> (dev_states, seed);

	dev_count_points <<<num_blocks, num_threads, 2 * num_vertices * sizeof(float) >>> (num_points, num_vertices, dev_vertices, num_iterations, dev_counts, length, dev_states);

	cudaError status = cudaMemcpy(counts, dev_counts, length * length * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(dev_vertices);
	cudaFree(dev_states);
	cudaFree(dev_counts);

	save_fractal(length, counts, "test.png");
	
	delete[] counts;

    return 0;
}




