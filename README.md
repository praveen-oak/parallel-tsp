# gpu-project
This project contains a sequential and parallel implementation of the 2-opt approximation algorithm to solve the Travelling Salesman Problem.

The sequential version is written in C and compiled on GCC.\
The parallel version is written in CUDA with compute capability of 3.5 or above.

Compilation
To compile the CPU code run:\
gcc -g -o seq_tsp seq_tsp.c -lm

To compile GPU code run:\
module load cuda-10.0\
nvcc -o gpu_tsp gpu_tsp.cu -arch=sm_35 -rdc=true



Running the code:\
For CPU\
./seq_tsp number_of_cities cities_coordinates_file optimal_cities_tour\
Eg:\
./seq_tsp 16 data/a16 data/a16_optimum


For GPU\
./gpu_tsp number_of_cities number_of_suggested_devices cities_coordinates_file
Eg:\
./gpu_tsp 16 20 data/a16

Note : \
Number of suggested devices is only a directive. The code detects the number of devices on the system and changes this accordingly.\
If the system has only 2 GPUs and input is to use 4, only 2 will be used\
The program is also limited to a max of 8 GPUs to run on irrespective of number of devices available. To change this, go into gpu_tsp.c and increase number of pthreads and pthread structures from 8.(Line 46 and 47)

Input Structure\
cities_coordinates_file\
This is a file containing the city number and its X and Y co ordinate.\
Format :\
city_number(int) x_coord(float) y_coord(float)\
Eg:\
4 23.56 34.97

The number of cities listed in file has to exactly match number of cities input in command line args.\
No other characters/new lines, special chars are allowed in file.

optimal_cities_tour\
This is a file containing the tour that results in the lowest cost for the city set.\
Format\
One number on each line representing the city.\
Eg : \
1\
2\
3

Note:\
The file should contain exactly the same number of cities as the cities input in command line args.\
The last city of the tour is inferred as the same as the first city in tour.\
Please look at the format used in data folder if you are confused. It contains sample datasets on which we tested and\ benchmarked the code


Data Folder\
This folder contains the datasets on which we ran the code to test and benchmark the results.\
For each dataset eg, if dataset is a666,\
a666 --> File with co ordinates\
a666_optimum --> File containing the optimal tour.

Results:\
The CPU program outputs the exact time in milliseconds the TSP part took, along with the cost of the solution tour and cost of the most optimal solution that exists for that dataset.

The CPU program outputs the exact time in milliseconds the TSP part took, along with the cost of the solution tour.
