#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include <stdbool.h>
//comment
#define index(i,j,cities)((i*cities)+j)

int two_opt(unsigned int *cycle, unsigned int *distance_array, unsigned int cities, unsigned int min_cost);
int travelling_salesman(int *distance_array, int cities, int *min_cycle);
int get_total_cost(unsigned int *cycle, unsigned int *distance_array, unsigned int cities);
void copy_array(unsigned int *first, unsigned int *second, int size);
void swap(unsigned int *cycle, int i, int j);

int main(int argc, char * argv[])
{
	if(argc != 3){
		fprintf(stderr, "usage: tsp cities city_distance_file\n");
	}
	FILE *fp;

	unsigned int cities = (unsigned int) atoi(argv[1]);
	fp=fopen(argv[2], "r");
	if(fp == NULL)
   	{
      fprintf(stderr, "Error in opening the file with path %s",argv[1]);
      exit(1);
   	}
   	unsigned int *distance_array = (unsigned int *)malloc(cities*cities*sizeof(unsigned int));
   		
   	int i;
   	int j;
   	unsigned int temp_var;
   	for(i = 0; i< cities; i++){
   		for(j = 0; j < cities; j++){
   			fscanf(fp,"%d",&temp_var);

   			distance_array[i*cities + j] = temp_var;
   		}
   	}

	printf("Data has been read.\n");
	unsigned int *cycle = (unsigned int *)malloc((cities+1)*sizeof(unsigned int));

	unsigned int *min_cycle = (unsigned int *)malloc(cities*sizeof(unsigned int *));
	int min_tour = 1000000;
	for(j = 0; j < cities; j++){
		int k = 0;
		while(k < cities){
			cycle[k] = (j+k) % cities;
			k++;
		}
		cycle[cities] = j;
		int min_cost = get_total_cost(cycle, distance_array, cities);
		while(1){
			int temp_min = two_opt(cycle, distance_array, cities, min_cost);
			if(temp_min >= min_cost){
				break;
			}else{
				min_cost = temp_min;
			}
		}
		if(min_cost < min_tour){
			min_tour = min_cost;
			int l;
			for(l = 0; l < cities + 1; l++){
				min_cycle[l] = cycle[l];
			}
		}
		for(i = 0; i < cities+1; i++){
			printf("%d ", cycle[i]);
		}
		printf("%d \n", min_cost);

	}
	for(i = 0; i < cities+1; i++){
		printf("%d ", min_cycle[i]);
	}

	printf(" %d \n", min_tour);
	
	
}

int get_total_cost(unsigned int *cycle, unsigned int *distance_array, unsigned int cities){
	int i = 0;
	int total_cost = 0;
	for(i = 1; i< cities+1; i++){
		total_cost = total_cost + distance_array[cycle[i-1]*cities + cycle[i]];
	}
	return total_cost;

}

void swap(unsigned int *cycle, int first, int second){
	int temp = cycle[first];
	cycle[first] = cycle[second];
	cycle[second] = temp;
}
int update_cycle(int *cycle, int i, int j){
	//reverse i to j
	while(i < j){
		swap(cycle, i, j);
		i++;
		j--;
	}
}

int two_opt(unsigned int *cycle, unsigned int *distance_array, unsigned int cities, unsigned int min_cost){

	int i;
	int j;
	int temp;
	int min_i;
	int min_j;
	int cost = 0;
	int total_cost;
	int temp_cost = min_cost;
	for(i = 1; i < cities; i++){
		for(j = i+1; j < cities; j++){
			//check if cost is reduced by swapping i and j
			temp_cost = min_cost + distance_array[i*cities+j] + distance_array[(j+1)*cities+i+1] - distance_array[(i)*cities+i+1] - distance_array[(j+1)*cities+j];
			update_cycle(cycle, i+1, j);
			total_cost = get_total_cost(cycle, distance_array, cities);
			
			if(total_cost < min_cost){
				return total_cost;
			}
			update_cycle(cycle, i+1, j);
		}
	}
	return min_cost;
}



