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


// void seq_two_opt(unsigned int *cycle, float *distance, float *cost_array, unsigned int cities){

// 	int cost_array_index;
// 	int city_i;
// 	int city_j;
// 	float temp;
// 	for(int i = 0; i < cities+1; i++){

// 		for(int j = 0; j < cities+1; j++){

// 			cost_array_index = i*(cities+1) + j;
// 			// if(i == 9 && j == 10){
// 			// 	// printf("Setting %f %d\n",cost_array[cost_array_index], cost_array_index);
// 			// 	printf("%f %d \n",distance[cycle[i]*cities + cycle[j+1]], cycle[i]*cities + cycle[j+1]);
// 			// 	printf("%f %d \n",distance[cycle[i-1]*cities + cycle[j]], cycle[i-1]*cities + cycle[j]);
// 			// 	printf("%f %d\n",distance[cycle[j]*cities + cycle[j+1]], cycle[j]*cities + cycle[j+1]);
// 			// 	printf("%f %d \n",distance[cycle[i-1]*cities + cycle[i]], cycle[i-1]*cities + cycle[i]);
// 			// }
// 			temp = cost_array[cost_array_index];
// 			temp = temp + distance[cycle[i]*cities + cycle[j+1]];
// 			temp = temp + distance[cycle[i-1]*cities + cycle[j]];
// 			temp = temp - distance[cycle[j]*cities + cycle[j+1]];
// 			temp = temp - distance[cycle[i-1]*cities + cycle[i]];
// 			// if(cost_array_index == 539){
// 			// 	printf("setting with %d %d %f %f \n",i,j,temp, cost_array[cost_array_index]);
// 			// }
// 			cost_array[cost_array_index] = temp;
// 		}
// 	}

// }




__global__ void two_opt(unsigned int *cycle, float *distance, unsigned int cities, float *min_val_array, int* min_i_array, int* min_j_array){
	float min_val = FLT_MAX;
	float temp_val;
	float min_i = -1;
	float min_j = -1;
	for(int i = blockIdx.x*blockDim.x + threadIdx.x+1; i < cities; i = i + blockDim.x*gridDim.x){
		for(int j = blockIdx.y*blockDim.y + threadIdx.y+1; j < cities; j = j + blockDim.y*gridDim.y){
			temp_val = distance[cycle[i]*cities + cycle[j+1]];
			temp_val += distance[cycle[i-1]*cities + cycle[j]];
			temp_val -= distance[cycle[j]*cities + cycle[j+1]];
			temp_val -= distance[cycle[i-1]*cities + cycle[i]];
			if(temp_val < min_val && i < j){
				min_val = temp_val;
				min_i = i;
				min_j = j;
			}
		}
	}
	int threadId = (blockIdx.x*blockDim.x + threadIdx.x)*blockDim.x*gridDim.x + (blockIdx.y*blockDim.y + threadIdx.y);
	min_val_array[threadId] = min_val;
	min_i_array[threadId] = min_i;
	min_j_array[threadId] = min_j;

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
			if(temp_val < min_val && i < j){
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

	int *cpu_min_i = (int *)malloc(total_threads*sizeof(int));
	int *gpu_min_i;
	CUDA_CALL(cudaMalloc(&gpu_min_i, total_threads*sizeof(int)));

	int *cpu_min_j = (int *)malloc(total_threads*sizeof(int));
	int *gpu_min_j;
	CUDA_CALL(cudaMalloc(&gpu_min_j, total_threads*sizeof(int)));
	

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

			CUDA_CALL(cudaMemset(gpu_min_val, 0, total_threads*sizeof(float)));
			CUDA_CALL(cudaMemset(gpu_min_i,0, total_threads*sizeof(int)));
			CUDA_CALL(cudaMemset(gpu_min_j, 0, total_threads*sizeof(int)));

			two_opt<<<gridDim, blockDim>>>(gpu_cycle, gpu_distance, cities, gpu_min_val, gpu_min_i, gpu_min_j);

			CUDA_CALL(cudaMemcpy(cpu_min_val, gpu_min_val, total_threads*sizeof(float), cudaMemcpyDeviceToHost));
			CUDA_CALL(cudaMemcpy(cpu_min_i, gpu_min_i, total_threads*sizeof(float), cudaMemcpyDeviceToHost));
			CUDA_CALL(cudaMemcpy(cpu_min_j, gpu_min_j, total_threads*sizeof(float), cudaMemcpyDeviceToHost));
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
				int min_i = cpu_min_i[min_index];
				int min_j = cpu_min_j[min_index];
				update_cycle(cpu_cycle, min_i, min_j);
			}
		}
	}
	printf("global minima = %f\n",global_minima);
}