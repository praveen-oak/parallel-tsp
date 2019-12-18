#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include <stdbool.h>
#include <float.h>
#include <string.h>
#include <math.h>
#include <time.h> 

//comment
#define index(i,j,cities)((i*cities)+j)

void read_data(unsigned int cities, float *x_coordinate, float *y_coordinate, FILE *fp);
void read_line(char *line, int index, float *x_coordinate, float *y_coordinate);
void get_distances(float *distance, float *x_coordinate, float *y_coordinate, unsigned int cities);
void read_optimum_file(FILE *fp, unsigned int *tour, unsigned int cities);
void read_files(FILE *fp, FILE *fp_optimum, float *distance, unsigned int *tour, unsigned int cities);
float error_from_optimal(float* distance_array, unsigned int *solution_cycle, unsigned int *optimal_cycle, unsigned int num_cities);
void write_results(char* filename, float optimal_cost, float solution_cost, float accuracy, float running_time);
void tsp_util(int num_cities, float* distances, unsigned int* optimal_tour);
float get_total_cost(unsigned int *cycle, float *distance_array, unsigned int num_cities);
void swap(unsigned int *cycle, int i, int j);
void update_cycle(unsigned int *cycle, int i, int j);
float two_opt(unsigned int *cycle, float *distance_array, unsigned int num_cities, float min_cost);

int main(int argc, char * argv[])
{
	if(argc != 4){
		fprintf(stderr, "usage: tsp cities city_distance_file optimum_tour_file\n");
	}
	unsigned int cities = (unsigned int) atoi(argv[1]);

	

	FILE *fp=fopen(argv[2], "r");
	FILE *fp_optimum=fopen(argv[3], "r");
	float *distance = (float *)malloc(cities*cities*sizeof(float));
	unsigned int *tour = (unsigned int *)malloc((cities+1)*sizeof(unsigned int *));
	read_files(fp, fp_optimum, distance, tour, cities);

	tsp_util(cities, distance, tour);

}

void read_files(FILE *fp, FILE *fp_optimum, float *distance, unsigned int *tour, unsigned int cities){

	float *x_coordinate = (float *)malloc(cities*sizeof(float));
	float *y_coordinate = (float *)malloc(cities*sizeof(float));
	if(fp == NULL){
      fprintf(stderr, "Error in opening the file");
      exit(1);
   	}
	read_data(cities, x_coordinate, y_coordinate, fp);
	// float *distance = (float *)malloc(cities*cities*sizeof(float));
	get_distances(distance, x_coordinate, y_coordinate, cities);

	int i;
	int j;
	
	if(fp == NULL){
      fprintf(stderr, "Error in opening optimum file");
      exit(1);
   	}
   	// unsigned int *tour = (unsigned int *)malloc((cities+1)*sizeof(unsigned int *));
   	read_optimum_file(fp_optimum, tour, cities);
}

void read_optimum_file(FILE *fp, unsigned int *tour, unsigned int cities){
	float total = 0;
	int i;
	int temp;
	for(i = 0; i < cities; i++){
		fscanf(fp,"%d",&temp);
		tour[i] = temp - 1;
	}
	tour[cities] = tour[0];

}

void get_distances(float *distance, float *x_coordinate, float *y_coordinate, unsigned int cities){
	int i = 0;
	int j = 0;
	float x_diff;
	float y_diff;
	for(i = 0; i < cities; i++){
		for(j = 0; j < cities; j++){
			x_diff = x_coordinate[i] - x_coordinate[j];
			y_diff = y_coordinate[i] - y_coordinate[j];
			distance[index(i,j,cities)] = sqrt(x_diff*x_diff + y_diff*y_diff);
		}
	}
	return;
}


void read_data(unsigned int cities, float *x_coordinate, float *y_coordinate, FILE *fp){
	
 
   	int i;
   	ssize_t read;
   	size_t len = 0;
   	char * line = NULL;
   	char *tuples;

   	for(i = 0; i < cities; i++){
   		read = getline(&line, &len, fp);
   		if(read == -1){
   			fprintf(stderr, "Error in reading file");
      		exit(1);
   		}
   		read_line(line, i, x_coordinate, y_coordinate);
   	}
   	return;
}

void read_line(char *line, int index, float *x_coordinate, float *y_coordinate){
	char *tuples;
	tuples = strtok(line," ");

	int temp = 0;
	while (tuples != NULL) {
    	if(temp == 1){
    		x_coordinate[index] = (float) atof(tuples);
    	}
    	if(temp == 2){
    		y_coordinate[index] = (float) atof(tuples);
    	}
    	tuples = strtok(NULL, " ");
    	temp++;
	}
	return;
}


struct Result{
	char* filename;
	float optimal_cost;
	float solution_cost;
	float accuracy;
	float time;
};

float error_from_optimal(float* distance_array, unsigned int *solution_cycle, unsigned int *optimal_cycle, unsigned int num_cities){
	float optimal_cost = get_total_cost(optimal_cycle, distance_array, num_cities);
	float solution_cost = get_total_cost(solution_cycle, distance_array, num_cities);
	float error = (solution_cost -  optimal_cost)/optimal_cost;
	float accuracy = 1-error;
	return accuracy;
}

void write_results(char* filename, float optimal_cost, float solution_cost, float accuracy, float running_time){
   FILE *fp;
   char dest_path[50];
   strcpy(dest_path,"results/");
   strcat(dest_path,filename);
   fp = fopen(dest_path, "w");
   fprintf(fp, "%s\t%f\t%f\t%f\t%f\n", filename, optimal_cost, solution_cost, accuracy, running_time);
   fclose(fp);
}

void tsp_util(int num_cities, float* distances, unsigned int* optimal_tour){
	unsigned int *cycle = (unsigned int *)malloc((num_cities+1)*sizeof(unsigned int));
	unsigned int *min_cycle = (unsigned int *)malloc((num_cities+1)*sizeof(unsigned int));
	int i, j, c, k;
	float global_min_tour = FLT_MAX, temp, local_min_tour; 

	struct timespec start, end;
	double time_usec = 0.0;

	clock_gettime(CLOCK_MONOTONIC, &start);
	for(j = 0; j < num_cities; j++){
		k=0;
		while(k < num_cities){
			cycle[k] = (j+k) % num_cities;
			k++;
		}
		cycle[num_cities] = j;
		local_min_tour = get_total_cost(cycle, distances, num_cities);
		while(1){
			temp = two_opt(cycle, distances, num_cities, local_min_tour);
			if(temp +.1 >= local_min_tour){
				break;
			}else{
				local_min_tour = temp;
			}
		}
		if(local_min_tour < global_min_tour){
			global_min_tour = local_min_tour;
			for(c = 0; c < num_cities + 1; c++){
				min_cycle[c] = cycle[c];
			}
		}
	}

	clock_gettime(CLOCK_MONOTONIC,&end);
	time_usec = (((double)end.tv_sec*1000 + (double)end.tv_nsec/1000000)-((double)start.tv_sec*1000 + (double)start.tv_nsec/1000000));
	printf("Time taken  = %lf milliseconds\n",time_usec);

	float optimal_cost = get_total_cost(optimal_tour, distances, num_cities);
	printf("Optimal cost: %f\n", optimal_cost);
	printf("Solution cost: %f\n", global_min_tour);
}


float get_total_cost(unsigned int *cycle, float *distance_array, unsigned int num_cities){
	int i = 0;
	float total_cost = 0;
	for(i = 1; i< num_cities+1; i++){
		total_cost = total_cost + distance_array[cycle[i-1]*num_cities + cycle[i]];
	}
	return total_cost;

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

float two_opt(unsigned int *cycle, float *distance_array, unsigned int num_cities, float min_cost){

	int i, j, k, min_i, min_j;
    float total_cost=min_cost,temp_cost;
	for(i = 1; i < num_cities-1; i++){
		for(j = i+1; j < num_cities; j++){

			//update_cycle(cycle,i,j);
			//total_cost = get_total_cost(cycle, distance_array, num_cities);
			temp_cost = min_cost + distance_array[index(cycle[i],cycle[j+1],num_cities)]+distance_array[index(cycle[i-1],cycle[j],num_cities)]-(distance_array[index(cycle[j],cycle[j+1],num_cities)]+distance_array[index(cycle[i-1],cycle[i],num_cities)]);
			if(temp_cost < total_cost){
				// return total_cost;
				total_cost =temp_cost;
				min_i=i;
				min_j=j;
			}
			//update_cycle(cycle,i,j);
		}
	}
	update_cycle(cycle,min_i,min_j);
	return total_cost;
}


