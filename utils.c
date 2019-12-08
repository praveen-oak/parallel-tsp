 #include <assert.h>

 #define CUDA_CALL(x) do {						\
   cudaError_t ____rc = (x);					\
   assert(____rc == cudaSuccess);					\
} while (0)

void swap(unsigned int *cycle, int i, int j);
void update_cycle(unsigned int *cycle, int i, int j);
float get_total_cost(unsigned int *cycle, float *distance_array, unsigned int num_cities);


float get_total_cost(unsigned int *cycle, float *distance_array, unsigned int num_cities){
	int i = 0;
	float total_cost = 0;
	for(i = 1; i< num_cities+1; i++){
		total_cost = total_cost + distance_array[index(cycle[i-1], cycle[i], num_cities)];
	}
	return total_cost;

}

void allocate_cycle(unsigned int *cycle, unsigned int start, unsigned int cities){
	int j;
	for(j = 0; j < cities; j++){
		cycle[j] = (j+start)%cities;
	}
	cycle[cities] = start;
}

void swap(unsigned int *cycle, int i, int j){
	unsigned int temp = cycle[i];
	cycle[i] = cycle[j];
	cycle[j] = temp;
}
void update_cycle(unsigned int *cycle, int i, int j){
	//reverse i to j
	while(i < j){
		swap(cycle, i, j);
		i++;
		j--;
	}
}

float get_min_cost(float *cost_array, unsigned int *cycle, unsigned int cities){

	int i;
	int j;
	float min_val = FLT_MAX;
	float cost_val;
	int min_i;
	int min_j;
	for(i = 1; i < cities; i++){
		for(j = 1; j < cities; j++){
			cost_val = cost_array[index(i,j,cities+1)];
			if(cost_val <= min_val){
				min_i = i;
				min_j = j;
				min_val = cost_val;
			}
		}
	}
	printf("%d %d \n", cycle[min_i], cycle[min_j]);
	return min_val;
}

int get_min_val(float *min_val_array,unsigned int threads){
	float global_min = FLT_MAX;
	int min_index;
	for(int i = 0; i < threads; i++){
		if(min_val_array[i] < global_min){
			global_min = min_val_array[i];
			min_index = i;
		}
	}
	return min_index;
}


void set_min_cost(float *cost_array, unsigned int cities, float value){
	int i;
	int j;
	for(i = 0; i < cities+1; i++){
		for(j = 0; j < cities+1; j++){
			cost_array[i*(cities+1) + j] = value;
		}
	}
}

void create_and_copy_float_data(float *gpu_pointer, float *cpu_pointer, int size){
	CUDA_CALL(cudaMalloc(&gpu_pointer, size));
	CUDA_CALL(cudaMemcpy(gpu_pointer, cpu_pointer, size, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(cpu_pointer, gpu_pointer, size, cudaMemcpyDeviceToHost));
}

void create_and_copy_int_data(unsigned int *gpu_pointer, unsigned int *cpu_pointer, int size){
	CUDA_CALL(cudaMalloc(&gpu_pointer, size));
	CUDA_CALL(cudaMemcpy(gpu_pointer, cpu_pointer, size, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(cpu_pointer, gpu_pointer, size, cudaMemcpyDeviceToHost));
}



