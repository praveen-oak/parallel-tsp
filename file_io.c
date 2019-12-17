#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include <stdbool.h>
#include <float.h>
#include <string.h>
#include <math.h>


#define index(i,j,cities)((i*cities)+j)



void read_data(unsigned int cities, float *x_coordinate, float *y_coordinate, FILE *fp);
void read_line(char *line, int index, float *x_coordinate, float *y_coordinate);
void get_distances(float *distance, float *x_coordinate, float *y_coordinate, unsigned int cities);
void read_optimum_file(FILE *fp, unsigned int *tour, unsigned int cities);
void read_files(FILE *fp, float *distance, unsigned int *tour, unsigned int cities);



void read_files(FILE *fp,float *distance, unsigned int *tour, unsigned int cities){

	float *x_coordinate = (float *)malloc(cities*sizeof(float));
	float *y_coordinate = (float *)malloc(cities*sizeof(float));
	if(fp == NULL){
      fprintf(stderr, "Error in opening the file");
      exit(1);
   	}
	read_data(cities, x_coordinate, y_coordinate, fp);
	// float *distance = (float *)malloc(cities*cities*sizeof(float));
	get_distances(distance, x_coordinate, y_coordinate, cities);
}

void read_optimum_file(FILE *fp, unsigned int *tour, unsigned int cities){
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



void print_min_cost(float *cost_array, unsigned int cities){

	int i;
	int j;
	for(i = 0; i < cities; i++){
		for(j = 0; j < cities; j++){
			printf("%0.1f ",cost_array[index(i,j,cities+1)]);
		}
		printf("\n");
	}
}

void print_cycle(unsigned int *cycle, unsigned int cities){
	int i;
	for(i = 0; i < cities+1; i++){
		printf("%d ",cycle[i]);
	}
	printf("\n");
}