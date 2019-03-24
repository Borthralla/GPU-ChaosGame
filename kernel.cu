
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <chrono>
#include "chaos_game.h"
#include "curand_kernel.h"

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

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
	/*
	const int arraySize = 5;
	const int a[arraySize] = { 1, 2, 3, 4, 5 };
	const int b[arraySize] = { 10, 20, 30, 40, 50 };
	int c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
		c[0], c[1], c[2], c[3], c[4]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}
	*/

	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

	int num_blocks = 28;
	int num_threads = 128;
	int length = 4000;
	int num_vertices = 9;
	int num_points = 400000000;
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



// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
