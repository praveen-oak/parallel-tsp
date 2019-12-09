#include "file_io.c"
#include "utils.c"
#include <assert.h>

//comment
#define index(i,j,cities)((i*cities)+j)
#define THREADS_X 16
#define THREADS_Y 16

#define BLOCKS_X 32
#define BLOCKS_Y 32
#define STREAMS 8



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

__global__ void two_opt(unsigned int *cycle, float *distance, unsigned int cities, float *min_val_array, unsigned int* min_index_array){
	__shared__ float temp_min[THREADS_Y*THREADS_X];
	__shared__ float temp_min_index[THREADS_Y*THREADS_X];
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


	//total threads in each block = blockDim.x*blockDim.y
	//id of thread in block = threadIdx.x*blockDim.x + threadIdx.y
	int tid =  threadIdx.x*blockDim.x + threadIdx.y;
	int bid = blockIdx.x*gridDim.x + blockIdx.y;

	temp_min[tid] = min_val;
	temp_min_index[tid] = min_index;

	for(unsigned int stride = 1; stride < blockDim.x*blockDim.y; stride*=2){
		__syncthreads();
		if(tid %(2*stride) == 0){
			if(temp_min[tid] > temp_min[tid+stride]){
				temp_min[tid] = temp_min[tid+stride];
				temp_min_index[tid] = temp_min_index[tid+stride];
			}
		}
	}
	
	min_index_array[bid] = temp_min_index[0];
	min_val_array[bid] = temp_min[0];
}

void tsp_serial(float *cpu_distance, float *gpu_distance, unsigned int *cpu_cycle, unsigned int *gpu_cycle, float *cpu_min_val, float *gpu_min_val, unsigned int *cpu_min_index, unsigned int *gpu_min_index, unsigned int cities, unsigned int i){
	dim3 gridDim(BLOCKS_X, BLOCKS_Y);
	dim3 blockDim(THREADS_X, THREADS_Y);
	int min_index;
	unsigned int cycle_size = (cities+1)*sizeof(unsigned int);
	float global_minima = FLT_MAX;

	allocate_cycle(cpu_cycle, i, cities);

	while(true){
		float temp_cost = get_total_cost(cpu_cycle, cpu_distance, cities);
		CUDA_CALL(cudaMemcpy(gpu_cycle, cpu_cycle, cycle_size, cudaMemcpyHostToDevice));
		two_opt<<<gridDim, blockDim>>>(gpu_cycle, gpu_distance, cities, gpu_min_val, gpu_min_index);

		CUDA_CALL(cudaMemcpy(cpu_min_val, gpu_min_val, BLOCKS_X*BLOCKS_Y*sizeof(float), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(cpu_min_index, gpu_min_index, BLOCKS_X*BLOCKS_Y*sizeof(float), cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
		//2-opt costs have been calculated

		min_index = get_min_val(cpu_min_val,BLOCKS_X*BLOCKS_Y);
		if(cpu_min_val[min_index] >= -0.01){
			if(global_minima > temp_cost){
				global_minima = temp_cost;
				// memcpy(global_optimal_cycle, cpu_cycle, cycle_size);
			}
			break;
		}
		else{
			int min_agg_index = cpu_min_index[min_index];
			update_cycle(cpu_cycle, min_agg_index/cities, min_agg_index%cities);
		}
	}
	printf("global minima = %f\n",global_minima);
}

void tsp(float *cpu_distance, unsigned int cities){

	dim3 gridDim(BLOCKS_X, BLOCKS_Y);
	dim3 blockDim(THREADS_X, THREADS_Y);

	//create and assign data to gpu distance array
	unsigned int distance_size = cities*cities*sizeof(float);
	unsigned int cycle_size = (cities+1)*sizeof(unsigned int);

	float *gpu_distance;
	CUDA_CALL(cudaMalloc(&gpu_distance, distance_size));
	CUDA_CALL(cudaMemcpy(gpu_distance, cpu_distance, distance_size, cudaMemcpyHostToDevice));
	


	float *cpu_min_val_pointer[STREAMS];
	float *gpu_min_val_pointer[STREAMS];
	unsigned int *gpu_min_index_pointer[STREAMS];
	unsigned int *cpu_min_index_pointer[STREAMS];
	unsigned int *gpu_cycle_pointer[STREAMS];
	unsigned int *cpu_cycle_pointer[STREAMS];

	float *cpu_min_val;
	float *gpu_min_val;
	unsigned int *cpu_min_index;
	unsigned int *gpu_min_index;
	unsigned int *cpu_cycle;
	unsigned int *gpu_cycle;

	for(int i = 0; i < STREAMS; i++){
		cpu_min_val = (float *)malloc(BLOCKS_X*BLOCKS_Y*sizeof(float));
		CUDA_CALL(cudaMalloc(&gpu_min_val, BLOCKS_X*BLOCKS_Y*sizeof(float)));
		cpu_min_val_pointer[i] = cpu_min_val;
		gpu_min_val_pointer[i] = gpu_min_val;
		cpu_min_index = (unsigned int *)malloc(BLOCKS_X*BLOCKS_Y*sizeof(unsigned int));
		CUDA_CALL(cudaMalloc(&gpu_min_index, BLOCKS_X*BLOCKS_Y*sizeof(unsigned int)));
		gpu_min_index_pointer[i] = gpu_min_index;
		cpu_min_index_pointer[i] = cpu_min_index;

		cpu_cycle = (unsigned int *)malloc(cycle_size);
		CUDA_CALL(cudaMalloc(&gpu_cycle, cycle_size));
		cpu_cycle_pointer[i] = cpu_cycle;
		gpu_cycle_pointer[i] = gpu_cycle;

	}
	



	tsp_serial(cpu_distance, gpu_distance, cpu_cycle_pointer[0], gpu_cycle_pointer[0], cpu_min_val_pointer[0], gpu_min_val_pointer[0],cpu_min_index_pointer[0], gpu_min_index_pointer[0], cities, 0);
}