#include "file_io.c"
#include "utils.c"
#include <assert.h>
#include <pthread.h>

//comment
#define index(i,j,cities)((i*cities)+j)
#define THREADS_X 16
#define THREADS_Y 16

#define BLOCKS_X 1
#define BLOCKS_Y 1

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

__global__ void two_opt(unsigned int *cycle, float *distance, unsigned int cities){
	__shared__ float temp_min[THREADS_Y*THREADS_X];
	__shared__ float temp_min_index[THREADS_Y*THREADS_X];
	float min_val = FLT_MAX;
	float temp_val;
	float min_index = -1;
	int tid =  threadIdx.x*blockDim.x + threadIdx.y;
	int l = 0;
	while(l < 10000){
		for(int i = blockIdx.x*blockDim.x + threadIdx.x+1; i < cities; i = i + blockDim.x*gridDim.x){
			for(int j = blockIdx.y*blockDim.y + threadIdx.y+1; j < cities; j = j + blockDim.y*gridDim.y){
				temp_val = distance[cycle[i]*cities + cycle[j+1]] + distance[cycle[i-1]*cities + cycle[j]]-distance[cycle[j]*cities + cycle[j+1]]-distance[cycle[i-1]*cities + cycle[i]];
				if(temp_val < min_val && i < j){
					min_val = temp_val;
					min_index = i*cities+j;
				}
			}
		}
		

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
		__syncthreads();
		//we have the min values now
		//swap values
		if(temp_min[0] > -1){
			return;
		}
		if(tid == 0){
			int min_i = temp_min[0]/cities;
			int min_j = temp_min[0] - min_i*cities;
			int k = 0;
			unsigned int temp;
			for(k = 0; k <= (min_j-min_i)/2; k = k + THREADS_Y*THREADS_X){
				temp = cycle[min_i+tid+k];
				cycle[min_i+tid+k] = cycle[min_i-tid-k];
				cycle[min_i-tid-k] = cycle[min_i+tid+k];
			}
		}
		l++;
	}

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
	

	unsigned int cycle_size = (cities+1)*sizeof(unsigned int);
	unsigned int *cpu_cycle = (unsigned int *)malloc(cycle_size);
	unsigned int *global_optimal_cycle = (unsigned int *)malloc(cycle_size);
	unsigned int *gpu_cycle;
	CUDA_CALL(cudaMalloc(&gpu_cycle, cycle_size));

	float global_minima = FLT_MAX;
	for(int i = device_index; i < cities; i = i + devices){
		allocate_cycle(cpu_cycle, i, cities);
		CUDA_CALL(cudaMemcpy(gpu_cycle, cpu_cycle, cycle_size, cudaMemcpyHostToDevice));
		
		two_opt<<<gridDim, blockDim>>>(gpu_cycle, gpu_distance, cities);
		CUDA_CALL(cudaMemcpy(cpu_cycle, gpu_cycle, cycle_size, cudaMemcpyDeviceToHost));
		cudaDeviceSynchronize();
		float temp_cost = get_total_cost(cpu_cycle, cpu_distance, cities);
		if(global_minima > temp_cost){
			global_minima = temp_cost;
			memcpy(global_optimal_cycle, cpu_cycle, cycle_size);
		}
	}
	return_pointer[0] = global_minima;
	cudaFree(gpu_cycle);

	free(cpu_cycle);
    return NULL;
	
}