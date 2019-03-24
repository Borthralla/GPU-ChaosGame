#pragma once
#include <random>

struct float_point {
	float x;
	float y;
};

struct int_point {
	int x;
	int y;
};

float * make_vertices(int length, int num_vertices); // Allocates list of floats corresponding to x/y coordinates of vertices on square of side-length length with center length/2, length/2

struct int_point make_point(int num_vertices, float * vertices, float center, int num_iterations, std::default_random_engine &r);

int * count_points(int num_points, int num_vertices, int length, int num_iterations, std::default_random_engine &r);

void save_fractal(int length, int * counts, char filename[]);



