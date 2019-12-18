#include "file_io.c"
#include "utils.c"
#include <assert.h>
#include <pthread.h>
#include <time.h> 

#define THREADS_X 16
#define THREADS_Y 16

#define BLOCKS_X 16
#define BLOCKS_Y 16


int devices = 0;

//structure to send to each pthread
struct arg_struct {
    float *cpu_distance;
    float *gpu_distance;
    unsigned int cities;
    float *return_pointer;
    int device_index;
};

//function to be called by each thread
void *tsp(void *arguements);


int main(int argc, char * argv[])
{
	if(argc != 4){
		fprintf(stderr, "usage: gpu_tsp cities max_devices city_distance_file\n");
	}
	unsigned int cities = (unsigned int) atoi(argv[1]);
	unsigned int suggested_devices = (unsigned int) atoi(argv[2]);

	if(suggested_devices == 0){
		suggested_devices = cities/100;
	}

	unsigned int distance_array_size = cities*cities*sizeof(float);
	FILE *fp=fopen(argv[3], "r");

	//read the data from the file
	float *distance = (float *)malloc(distance_array_size);
	unsigned int *tour = (unsigned int *)malloc((cities+1)*sizeof(unsigned int *));
	read_files(fp, distance, tour, cities);	
	fclose(fp);

	//create thread structures
	struct arg_struct args[8];
	pthread_t threads[8];

	//get number of devices available
	CUDA_CALL(cudaGetDeviceCount (&devices));
	float *gpu_distance;

	//if suggested devices is lesser than max devices available, then change it
	if(suggested_devices < devices){
		devices = suggested_devices;
	}

	//we are using a max of 8 pthreads, and each pthread is mapped to a device, so code can accomadate a max of 8 devices
	if(devices > 8){
		devices = 8;
	}

	struct timespec start, end;
	double time_usec = 0.0;

	clock_gettime(CLOCK_MONOTONIC, &start);
	//run a thread for each device
	for(int i = 0; i < devices; i++){

		CUDA_CALL(cudaSetDevice(i));
		
		CUDA_CALL(cudaMalloc(&gpu_distance, distance_array_size));
		CUDA_CALL(cudaMemcpy(gpu_distance, distance, distance_array_size, cudaMemcpyHostToDevice));

		args[i].cpu_distance = distance;
		args[i].cities = cities;
		args[i].gpu_distance= gpu_distance;
		args[i].return_pointer = (float *)malloc(sizeof(float));
		args[i].device_index = i;
		if (pthread_create(&threads[i], NULL, &tsp, (void *)&args[i]) != 0) {
        	printf("Error in creating threads, exiting program\n");
        	return -1;
    	}
    	
	}

	//wait for threads
	for(int i = 0; i < devices; i++){
		pthread_join(threads[i], NULL);	
	}

	//get global minimum based on value returned by each thread
	float min_val = FLT_MAX;
	for(int i = 0; i < devices; i++){
		if(args[i].return_pointer[0] < min_val){
			min_val = args[i].return_pointer[0];
		}
	}
	clock_gettime(CLOCK_MONOTONIC,&end);
	time_usec = (((double)end.tv_sec*1000 + (double)end.tv_nsec/1000000)-((double)start.tv_sec*1000 + (double)start.tv_nsec/1000000));

	printf("Time taken = %lf milliseconds\n",time_usec);

	printf("Global minimum value is %f\n",min_val);
	free(distance);
	free(tour);
	cudaFree(gpu_distance);

}

//GPU kernel that runs 2-opt algo in parallel
__global__ void two_opt(unsigned int *cycle, float *distance, unsigned int cities, float *min_val_array, unsigned int* min_index_array){
	//shared array to store most negetive decrease reported by each thread
	__shared__ float temp_min[THREADS_Y*THREADS_X];
	__shared__ float temp_min_index[THREADS_Y*THREADS_X];
	float min_val = FLT_MAX;
	float temp_val;
	float min_index = -1;
	for(int i = blockIdx.x*blockDim.x + threadIdx.x+1; i < cities; i = i + blockDim.x*gridDim.x){
		for(int j = blockIdx.y*blockDim.y + threadIdx.y+1; j < cities; j = j + blockDim.y*gridDim.y){
			temp_val = distance[cycle[i]*cities + cycle[j+1]]+distance[cycle[i-1]*cities + cycle[j]]-distance[cycle[j]*cities + cycle[j+1]]-distance[cycle[i-1]*cities + cycle[i]];
			if(temp_val < min_val && i < j){
				min_val = temp_val;
				min_index = i*cities+j; //this is being done to save space, both i and j values can be stored and retrieved this way
			}
		}
	}


	//total threads in each block = blockDim.x*blockDim.y
	//id of thread in block = threadIdx.x*blockDim.x + threadIdx.y
	int tid =  threadIdx.x*blockDim.x + threadIdx.y;
	int bid = blockIdx.x*gridDim.x + blockIdx.y;

	temp_min[tid] = min_val;
	temp_min_index[tid] = min_index;

	//now reduce the min array so that the index = 0 in array has the min value and index
	for(unsigned int stride = 1; stride < blockDim.x*blockDim.y; stride*=2){
		__syncthreads();
		if(tid %(2*stride) == 0){
			if(temp_min[tid] > temp_min[tid+stride]){
				temp_min[tid] = temp_min[tid+stride];
				temp_min_index[tid] = temp_min_index[tid+stride];
			}
		}
	}
	
	//save the min value for the block into global index for the block
	if(tid == 0){
		min_index_array[bid] = temp_min_index[0];	
		min_val_array[bid] = temp_min[0];
	}
	
	
}

void *tsp(void *arguments){

	struct arg_struct *args = (struct arg_struct *)arguments;


	float *gpu_distance = args -> gpu_distance;
	float *cpu_distance = args -> cpu_distance; 
	unsigned int cities = args -> cities;
	float *return_pointer = args -> return_pointer;
	int device_index = args -> device_index;

	//set the device to run on based on what was sent by master thread
	CUDA_CALL(cudaSetDevice(device_index));

	dim3 gridDim(BLOCKS_X, BLOCKS_Y);
	dim3 blockDim(THREADS_X, THREADS_Y);
	int min_index;

	//stores the min value reported by each block
	float *cpu_min_val = (float *)malloc(BLOCKS_X*BLOCKS_Y*sizeof(float));
	float *gpu_min_val;
	CUDA_CALL(cudaMalloc(&gpu_min_val, BLOCKS_X*BLOCKS_Y*sizeof(float)));

	//stored the i and j values(as i*cities + j) of the swap that will result in the min value
	unsigned int *cpu_min_index = (unsigned int *)malloc(BLOCKS_X*BLOCKS_Y*sizeof(unsigned int));
	unsigned int *gpu_min_index;
	CUDA_CALL(cudaMalloc(&gpu_min_index, BLOCKS_X*BLOCKS_Y*sizeof(unsigned int)));
	
	//stores current cycle
	unsigned int cycle_size = (cities+1)*sizeof(unsigned int);
	unsigned int *cpu_cycle = (unsigned int *)malloc(cycle_size);
	unsigned int *global_optimal_cycle = (unsigned int *)malloc(cycle_size);
	unsigned int *gpu_cycle;
	CUDA_CALL(cudaMalloc(&gpu_cycle, cycle_size));

	//run 8 streams
	const int num_streams = 8;
  	cudaStream_t streams[num_streams];
  	cudaStream_t current_stream;
  	int stream_index = 0;
  	for (int i = 0; i < num_streams; i++) {
    	cudaStreamCreate(&streams[i]);
  	}

	float global_minima = FLT_MAX;
	for(int i = device_index; i < cities; i = i + devices){

		//allocate initial cities cycle
		allocate_cycle(cpu_cycle, i, cities);
		current_stream = streams[stream_index%num_streams];
		while(true){
			//get current cost
			float temp_cost = get_total_cost(cpu_cycle, cpu_distance, cities);
			//move current cycle to gpu and find best swap
			CUDA_CALL(cudaMemcpyAsync(gpu_cycle, cpu_cycle, cycle_size, cudaMemcpyHostToDevice, current_stream));
			two_opt<<<gridDim, blockDim, 0, current_stream>>>(gpu_cycle, gpu_distance, cities, gpu_min_val, gpu_min_index);

			//move best reported swap and most decrease of that swap to CPU
			CUDA_CALL(cudaMemcpyAsync(cpu_min_val, gpu_min_val, BLOCKS_X*BLOCKS_Y*sizeof(float), cudaMemcpyDeviceToHost,current_stream));
			CUDA_CALL(cudaMemcpyAsync(cpu_min_index, gpu_min_index, BLOCKS_X*BLOCKS_Y*sizeof(int), cudaMemcpyDeviceToHost,current_stream));
			cudaStreamSynchronize(current_stream);

			//using this, calculated best values across blocks
			min_index = get_min_val(cpu_min_val,BLOCKS_X*BLOCKS_Y);

			//if the benefit of best swap is less than an increase of 0.1, we are close to minima, can exit
			if(cpu_min_val[min_index] >= -.1){
				if(global_minima > temp_cost){
					global_minima = temp_cost;
					memcpy(global_optimal_cycle, cpu_cycle, cycle_size);
				}
				break;
			}
			else{
				//otherwise find best indices for the swap and update the cycle of cities
				int min_agg_index = cpu_min_index[min_index];
				update_cycle(cpu_cycle, min_agg_index/cities, min_agg_index%cities);
			}
		}
		stream_index++;
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