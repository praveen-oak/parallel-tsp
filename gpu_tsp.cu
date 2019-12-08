#include "file_io.c"
#include "utils.c"
#include <assert.h>

//comment
#define index(i,j,cities)((i*cities)+j)



void tsp(float *distance, unsigned int cities);
int main(int argc, char * argv[])
{
	if(argc != 4){
		fprintf(stderr, "usage: tsp cities city_distance_file optimum_tour_file\n");
	}
	unsigned int cities = (unsigned int) atoi(argv[1]);

	unsigned int distance_array_size = cities*cities*sizeof(float);
	FILE *fp=fopen(argv[2], "r");
	FILE *fp_optimum=fopen(argv[3], "r");

	float *distance = (float *)malloc(distance_array_size);
	unsigned int *tour = (unsigned int *)malloc((cities+1)*sizeof(unsigned int *));
	read_files(fp, fp_optimum, distance, tour, cities);	
	tsp(distance, cities);
}





__global__ void two_opt(unsigned int *cycle, float *distance, float *cost_array, unsigned int cities){
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	int j = blockIdx.y*blockDim.y + threadIdx.y;
	int i_stride = blockDim.x*gridDim.x;
	int j_stride = blockDim.y*gridDim.y;

	unsigned int city_i = cycle[i];
	unsigned int city_j = cycle[j];
	unsigned int cost_array_index = index(city_i, city_j, cities);


	for(int i = blockIdx.x*blockDim.x + threadIdx.x+1; i < cities; i = i + i_stride){
		for(int j = blockIdx.y*blockDim.y + threadIdx.y+1; j < cities; j = j + j_stride){
			cost_array[cost_array_index] += distance[index(cycle[i], cycle[j+1], cities)];
			cost_array[cost_array_index] += distance[index(cycle[i-1], cycle[j], cities)];
			cost_array[cost_array_index] -= distance[index(cycle[j], cycle[j+1], cities)];
			cost_array[cost_array_index] -= distance[index(cycle[i-1], cycle[i], cities)];
		}
	}
}

__global__ void find_min(float *cost_array, unsigned int cities, float *min_val_array, int* min_i_array, int* min_j_array){

	int i_stride = blockDim.x*gridDim.x;
	int j_stride = blockDim.y*gridDim.y;
	int threadId = (blockIdx.x*blockDim.x + threadIdx.x)*i_stride + (blockIdx.y*blockDim.y + threadIdx.y);
	float min_val = FLT_MAX;
	float temp_val = FLT_MAX;
	int min_i;
	int min_j;
	for(int i = blockIdx.x*blockDim.x + threadIdx.x+1; i < cities; i = i + i_stride){
		for(int j = blockIdx.y*blockDim.y + threadIdx.y+1; j < cities; j = j + j_stride){
			temp_val = cost_array[i*(cities+1)+j];
			if(temp_val < min_val){
				min_i = i;
				min_j = j;
				min_val = temp_val;
			}
		}
	}
	min_val_array[threadId] = min_val;
	min_i_array[threadId] = min_i;
	min_j_array[threadId] = min_j;
}


void tsp(float *cpu_distance, unsigned int cities){
	int i;
	int j;

	//create and assign data to gpu distance array
	unsigned int distance_size = cities*cities*sizeof(float);
	// printf("%d \n",distance_size);
	float *gpu_distance;
	
	CUDA_CALL(cudaMalloc(&gpu_distance, distance_size));
	CUDA_CALL(cudaMemcpy(gpu_distance, cpu_distance, distance_size, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(cpu_distance, gpu_distance, distance_size, cudaMemcpyDeviceToHost));

	
	unsigned int cycle_size = (cities+1)*sizeof(unsigned int);
	unsigned int *cpu_cycle = (unsigned int *)malloc(cycle_size);
	allocate_cycle(cpu_cycle, 0, cities);
	unsigned int *gpu_cycle;
	CUDA_CALL(cudaMalloc(&gpu_cycle, cycle_size));
	CUDA_CALL(cudaMemcpy(gpu_cycle, cpu_cycle, cycle_size, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(cpu_cycle, gpu_cycle, cycle_size, cudaMemcpyDeviceToHost));

	float temp_cost = get_total_cost(cpu_cycle, cpu_distance, cities);
	unsigned int cost_size = (cities+1)*(cities+1)*sizeof(float);
	float *cpu_cost = (float *)malloc(cost_size); 
	set_min_cost(cpu_cost, cities, temp_cost);

	float *gpu_cost;
	CUDA_CALL(cudaMalloc(&gpu_cost, cost_size));
	CUDA_CALL(cudaMemcpy(gpu_cost, cpu_cost, cost_size, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(cpu_cost, gpu_cost, cost_size, cudaMemcpyDeviceToHost));

	
	cudaDeviceSynchronize();
	
	dim3 gridDim(16, 16);
	dim3 blockDim(16, 16);
	int total_threads = blockDim.x*blockDim.y*gridDim.x*gridDim.y;

	two_opt<<<gridDim, blockDim>>>(gpu_cycle, gpu_distance, gpu_cost, cities);
	CUDA_CALL(cudaMemcpy(cpu_cost, gpu_cost, cost_size, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	//2-opt costs have been calculated


	float *cpu_min_val = (float *)malloc(total_threads*sizeof(float));
	float *gpu_min_val;
	CUDA_CALL(cudaMalloc(&gpu_min_val, total_threads*sizeof(float)));

	int *cpu_min_i = (int *)malloc(total_threads*sizeof(int));
	int *gpu_min_i;
	CUDA_CALL(cudaMalloc(&gpu_min_i, total_threads*sizeof(int)));

	int *cpu_min_j = (int *)malloc(total_threads*sizeof(int));
	int *gpu_min_j;
	CUDA_CALL(cudaMalloc(&gpu_min_j, total_threads*sizeof(int)));

	CUDA_CALL(cudaMemset(gpu_min_val, 0, total_threads*sizeof(float)));
	CUDA_CALL(cudaMemset(gpu_min_i,0, total_threads*sizeof(int)));
	CUDA_CALL(cudaMemset(gpu_min_j, 0, total_threads*sizeof(int)));
	memset(cpu_min_val, 0, total_threads*sizeof(float));
	memset(cpu_min_i, 0, total_threads*sizeof(int));
	memset(cpu_min_j, 0, total_threads*sizeof(int));


	cudaDeviceSynchronize();

	find_min<<<gridDim, blockDim>>>(gpu_cost, cities, gpu_min_val, gpu_min_i, gpu_min_j);
	cudaDeviceSynchronize();
	CUDA_CALL(cudaMemcpy(cpu_min_val, gpu_min_val, total_threads*sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(cpu_min_i, gpu_min_i, total_threads*sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaMemcpy(cpu_min_j, gpu_min_j, total_threads*sizeof(float), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();

	int min_index = get_min_val(cpu_min_val,total_threads);
	int min_i = cpu_min_i[min_index];
	int min_j = cpu_min_j[min_index];
	printf("to swap = %d %d \n",min_i, min_j);
	update_cycle(cpu_cycle, min_i, min_j);

}