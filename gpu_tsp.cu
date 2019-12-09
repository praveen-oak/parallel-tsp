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

__global__ void two_opt(unsigned int *cycle, float *distance, unsigned int cities, float *min_val_array, int* min_index_array){
	float min_val = FLT_MAX;
	float temp_val;
	float min_index = -1;
	for(int i = blockIdx.x*blockDim.x + threadIdx.x+1; i < cities; i = i + blockDim.x*gridDim.x){
		for(int j = blockIdx.y*blockDim.y + threadIdx.y+1; j < cities; j = j + blockDim.y*gridDim.y){
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
	int threadId = (blockIdx.x*blockDim.x + threadIdx.x)*blockDim.x*gridDim.x + (blockIdx.y*blockDim.y + threadIdx.y);
	min_val_array[threadId] = min_val;
	min_index_array[threadId] = min_index;
}


void tsp(float *cpu_distance, unsigned int cities){

	dim3 gridDim(16, 16);
	dim3 blockDim(16, 16);
	int total_threads = blockDim.x*blockDim.y*gridDim.x*gridDim.y;

	//create and assign data to gpu distance array
	unsigned int distance_size = cities*cities*sizeof(float);
	float *gpu_distance;
	int min_index;


	CUDA_CALL(cudaMalloc(&gpu_distance, distance_size));
	CUDA_CALL(cudaMemcpy(gpu_distance, cpu_distance, distance_size, cudaMemcpyHostToDevice));
	
	float *cpu_min_val = (float *)malloc(total_threads*sizeof(float));
	float *gpu_min_val;
	CUDA_CALL(cudaMalloc(&gpu_min_val, total_threads*sizeof(float)));

	int *cpu_min_index = (int *)malloc(total_threads*sizeof(int));
	int *gpu_min_index;
	CUDA_CALL(cudaMalloc(&gpu_min_index, total_threads*sizeof(int)));
	

	unsigned int cycle_size = (cities+1)*sizeof(unsigned int);
	unsigned int *cpu_cycle = (unsigned int *)malloc(cycle_size);
	unsigned int *gpu_cycle;
	CUDA_CALL(cudaMalloc(&gpu_cycle, cycle_size));

	float global_minima = FLT_MAX;
	for(int i = 0; i< cities; i++){
		allocate_cycle(cpu_cycle, i, cities);

		while(true){
			float temp_cost = get_total_cost(cpu_cycle, cpu_distance, cities);
			CUDA_CALL(cudaMemcpy(gpu_cycle, cpu_cycle, cycle_size, cudaMemcpyHostToDevice));
			two_opt<<<gridDim, blockDim>>>(gpu_cycle, gpu_distance, cities, gpu_min_val, gpu_min_index);

			CUDA_CALL(cudaMemcpy(cpu_min_val, gpu_min_val, total_threads*sizeof(float), cudaMemcpyDeviceToHost));
			CUDA_CALL(cudaMemcpy(cpu_min_index, gpu_min_index, total_threads*sizeof(float), cudaMemcpyDeviceToHost));
			cudaDeviceSynchronize();
			//2-opt costs have been calculated

			min_index = get_min_val(cpu_min_val,total_threads);
			if(cpu_min_val[min_index] >= -0.001){
				if(global_minima > temp_cost){
					global_minima = temp_cost;
				}
				break;
			}
			else{
				// printf("%f \n",)
				int min_agg_index = cpu_min_index[min_index];
				update_cycle(cpu_cycle, min_agg_index/cities, min_agg_index%cities);
			}
		}
	}
	printf("global minima = %f\n",global_minima);
}