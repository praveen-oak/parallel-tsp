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
	allocate_cycle(cpu_cycle, 20, cities);
	unsigned int *gpu_cycle;
	CUDA_CALL(cudaMalloc(&gpu_cycle, cycle_size));
	CUDA_CALL(cudaMemcpy(gpu_cycle, cpu_cycle, cycle_size, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(cpu_cycle, gpu_cycle, cycle_size, cudaMemcpyDeviceToHost));

	float temp_cost = get_total_cost(cpu_cycle, cpu_distance, cities);
	unsigned int cost_size = (cities+1)*(cities+1)*sizeof(float);
	float *cpu_cost = (float *)malloc(cost_size); 
	set_min_cost(cpu_cost, cities, temp_cost);

	for(i = 0; i < cities+1; i++){
		for(j = 0; j< cities+1; j++){
			if(cpu_cost[i*(cities+1)+j] != temp_cost){
				printf("%d %d\n",i,j);
			}
		}
	}

	float *gpu_cost;
	CUDA_CALL(cudaMalloc(&gpu_cost, cost_size));
	CUDA_CALL(cudaMemcpy(gpu_cost, cpu_cost, cost_size, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(cpu_cost, gpu_cost, cost_size, cudaMemcpyDeviceToHost));

	
	cudaDeviceSynchronize();
	dim3 gridDim(16, 16);
	dim3 blockDim(16, 16);

	two_opt<<<gridDim, blockDim>>>(gpu_cycle, gpu_distance, gpu_cost, cities);
	// get_min_val<<<gridDim, blockDim>>>(gpu_cost, cities);
	CUDA_CALL(cudaMemcpy(cpu_cost, gpu_cost, cost_size, cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	

	for(i = 0; i < cities+1; i++){
		for(j = 0; j< cities+1; j++){
			if(cpu_cost[i*(cities+1)+j] != cpu_cost[i*(cities+1)+j]){
				printf("%d %d\n",i,j);
			}
		}
	}
	float min_cost = get_min_cost(cpu_cost, cpu_cycle, cities);
	printf("%f \n",min_cost);

}