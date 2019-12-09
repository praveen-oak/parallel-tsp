#include "file_io.c"
#include "utils.c"
#include <assert.h>
#include <pthread.h>

//comment
#define index(i,j,cities)((i*cities)+j)
#define THREADS_X 32
#define THREADS_Y 32

#define BLOCKS_X 32
#define BLOCKS_Y 32

int devices = 0;
struct arg_struct {
    float *cpu_distance;
    float *gpu_distance;
    unsigned int cities;
    float *return_pointer;
    int device_index;
};

void *tsp(void *arguements);
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
	fclose(fp);
	fclose(fp_optimum);

	struct arg_struct args[8];
	pthread_t stream_threads[8];

	CUDA_CALL(cudaGetDeviceCount (&devices));
	float *gpu_distance;


	for(int i = 0; i < devices; i++){

		CUDA_CALL(cudaSetDevice(i));
		
		CUDA_CALL(cudaMalloc(&gpu_distance, distance_array_size));
		CUDA_CALL(cudaMemcpy(gpu_distance, distance, distance_array_size, cudaMemcpyHostToDevice));
		cudaDeviceSynchronize();

		args[i].cpu_distance = distance;
		args[i].cities = cities;
		args[i].gpu_distance= gpu_distance;
		args[i].return_pointer = (float *)malloc(sizeof(float));
		args[i].device_index = i;
		if (pthread_create(&stream_threads[i], NULL, &tsp, (void *)&args[i]) != 0) {
        	printf("Error in creating threads, exiting program\n");
        	return -1;
    	}
    	
	}
	for(int i = 0; i < devices; i++){
		pthread_join(stream_threads[i], NULL);	
	}
	float min_val = FLT_MAX;
	for(int i = 0; i < devices; i++){
		if(args[i].return_pointer[0] < min_val){
			min_val = args[i].return_pointer[0];
		}
	}
	printf("Global minimum value is %f\n",min_val);
	free(distance);
	free(tour);
	cudaFree(gpu_distance);

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

void *tsp(void *arguments){

	struct arg_struct *args = (struct arg_struct *)arguments;


	float *gpu_distance = args -> gpu_distance;
	float *cpu_distance = args -> cpu_distance; 
	unsigned int cities = args -> cities;
	float *return_pointer = args -> return_pointer;
	int device_index = args -> device_index;
	CUDA_CALL(cudaSetDevice(device_index));

	dim3 gridDim(BLOCKS_X, BLOCKS_Y);
	dim3 blockDim(THREADS_X, THREADS_Y);
	int min_index;
	float *cpu_min_val = (float *)malloc(BLOCKS_X*BLOCKS_Y*sizeof(float));
	float *gpu_min_val;
	CUDA_CALL(cudaMalloc(&gpu_min_val, BLOCKS_X*BLOCKS_Y*sizeof(float)));

	unsigned int *cpu_min_index = (unsigned int *)malloc(BLOCKS_X*BLOCKS_Y*sizeof(unsigned int));
	unsigned int *gpu_min_index;
	CUDA_CALL(cudaMalloc(&gpu_min_index, BLOCKS_X*BLOCKS_Y*sizeof(unsigned int)));
	

	unsigned int cycle_size = (cities+1)*sizeof(unsigned int);
	unsigned int *cpu_cycle = (unsigned int *)malloc(cycle_size);
	unsigned int *global_optimal_cycle = (unsigned int *)malloc(cycle_size);
	unsigned int *gpu_cycle;
	CUDA_CALL(cudaMalloc(&gpu_cycle, cycle_size));

	float global_minima = FLT_MAX;
	for(int i = device_index; i < cities; i = i + devices){
		allocate_cycle(cpu_cycle, i, cities);

		while(true){
			float temp_cost = get_total_cost(cpu_cycle, cpu_distance, cities);
			CUDA_CALL(cudaMemcpy(gpu_cycle, cpu_cycle, cycle_size, cudaMemcpyHostToDevice));
			two_opt<<<gridDim, blockDim>>>(gpu_cycle, gpu_distance, cities, gpu_min_val, gpu_min_index);

			CUDA_CALL(cudaMemcpy(cpu_min_val, gpu_min_val, BLOCKS_X*BLOCKS_Y*sizeof(float), cudaMemcpyDeviceToHost));
			CUDA_CALL(cudaMemcpy(cpu_min_index, gpu_min_index, BLOCKS_X*BLOCKS_Y*sizeof(float), cudaMemcpyDeviceToHost));
			cudaDeviceSynchronize();

			min_index = get_min_val(cpu_min_val,BLOCKS_X*BLOCKS_Y);
			if(cpu_min_val[min_index] >= -0.001){
				if(global_minima > temp_cost){
					global_minima = temp_cost;
					memcpy(global_optimal_cycle, cpu_cycle, cycle_size);
				}
				break;
			}
			else{
				int min_agg_index = cpu_min_index[min_index];
				update_cycle(cpu_cycle, min_agg_index/cities, min_agg_index%cities);
			}
		}
	}
	return_pointer[0] = global_minima;

	cudaFree(gpu_min_val);
	cudaFree(gpu_min_index);
	cudaFree(gpu_cycle);

	free(cpu_cycle);
	free(cpu_min_val);
	free(cpu_min_index);

    return NULL;
	
}