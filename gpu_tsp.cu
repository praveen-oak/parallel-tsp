#include "file_io.c"
#include "utils.c"
#include <assert.h>

//comment
#define index(i,j,cities)((i*cities)+j)
#define THREADS_X 32
#define THREADS_Y 32

#define BLOCKS_X 1
#define BLOCKS_Y 1



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

__global__ void two_opt(unsigned int *cycle, float *distance, unsigned int cities, float *final_min_val){
	__shared__ float temp_min[THREADS_Y*THREADS_X];
	__shared__ float temp_min_index[THREADS_Y*THREADS_X];
	float min_val = FLT_MAX;
	float temp_val;
	float min_index = -1;
	int i;
	int j;
	for(i = blockIdx.x*blockDim.x + threadIdx.x+1; i < cities; i = i + blockDim.x*gridDim.x){
		for(j = blockIdx.y*blockDim.y + threadIdx.y+1; j < cities; j = j + blockDim.y*gridDim.y){
			temp_val = distance[cycle[i]*cities + cycle[j+1]];
			temp_val += distance[cycle[i-1]*cities + cycle[j]];
			temp_val -= distance[cycle[j]*cities + cycle[j+1]];
			temp_val -= distance[cycle[i-1]*cities + cycle[i]];
			if(temp_val < min_val && i < j){
				min_val = temp_val;
				min_index = i*cities+j;
			}
		}
	}


	//total threads in each block = blockDim.x*blockDim.y
	//id of thread in block = threadIdx.x*blockDim.x + threadIdx.y
	int tid =  threadIdx.x*blockDim.x + threadIdx.y;
	temp_min_index[tid] = min_index;
	temp_min[tid] = min_val;  

	for(unsigned int stride = 1; stride < blockDim.x*blockDim.y; stride*=2){
		__syncthreads();
		if(tid %(2*stride) == 0){
			if(temp_min[tid] > temp_min[tid+stride]){
				temp_min[tid] = temp_min[tid+stride];
				temp_min_index[tid] = temp_min_index[tid+stride];
			}
		}
	}
	final_min_val[0] = temp_min[0];

	int temp;
	//only the first thread will do swaps
	if(tid == 0){
		i = temp_min_index[0]/cities;
		j = temp_min_index[0] - cities*i;
		while(i < j){
			temp = cycle[i];
			cycle[i] = cycle[j];
			cycle[j] = temp;
			i++;
			j--;
		}
	}
}


void tsp(float *cpu_distance, unsigned int cities){

	dim3 gridDim(BLOCKS_X, BLOCKS_Y);
	dim3 blockDim(THREADS_X, THREADS_Y);

	//create and assign data to gpu distance array
	unsigned int distance_size = cities*cities*sizeof(float);
	float *gpu_distance;

	CUDA_CALL(cudaMalloc(&gpu_distance, distance_size));
	CUDA_CALL(cudaMemcpy(gpu_distance, cpu_distance, distance_size, cudaMemcpyHostToDevice));
	
	float *cpu_min_val = (float *)malloc(sizeof(float));
	float *gpu_min_val;
	CUDA_CALL(cudaMalloc(&gpu_min_val, sizeof(float)));

	unsigned int cycle_size = (cities+1)*sizeof(unsigned int);
	unsigned int *cpu_cycle = (unsigned int *)malloc(cycle_size);
	unsigned int *global_optimal_cycle = (unsigned int *)malloc(cycle_size);
	unsigned int *gpu_cycle;
	CUDA_CALL(cudaMalloc(&gpu_cycle, cycle_size));

	float global_minima = FLT_MAX;
	float local_minima = FLT_MAX;
	for(int i = 0; i< cities; i++){
		allocate_cycle(cpu_cycle, i, cities);
		CUDA_CALL(cudaMemcpy(gpu_cycle, cpu_cycle, cycle_size, cudaMemcpyHostToDevice));
		while(true){
			// float temp_cost = get_total_cost(cpu_cycle, cpu_distance, cities);
			two_opt<<<gridDim, blockDim>>>(gpu_cycle, gpu_distance, cities, gpu_min_val);
			CUDA_CALL(cudaMemcpy(cpu_min_val, gpu_min_val, sizeof(float), cudaMemcpyDeviceToHost));
			// CUDA_CALL(cudaMemcpy(cpu_min_index, gpu_min_index, BLOCKS_X*BLOCKS_Y*sizeof(float), cudaMemcpyDeviceToHost));
			cudaDeviceSynchronize();
			//2-opt costs have been calculated

			if(cpu_min_val[0] >= -0.01){
				CUDA_CALL(cudaMemcpy(cpu_cycle, gpu_cycle, cycle_size, cudaMemcpyDeviceToHost));
				local_minima = get_total_cost(cpu_cycle, cpu_distance, cities);
				if(global_minima > local_minima){
					global_minima = local_minima;
					memcpy(global_optimal_cycle, cpu_cycle, cycle_size);
				}
				break;
			}
		}
	}
	printf("global minima = %f\n",global_minima);
}