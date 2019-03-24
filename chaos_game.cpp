#define _USE_MATH_DEFINES
#include "chaos_game.h"
#include <functional>
#include <iostream>
#include <png.h>

float * make_vertices(int length, int num_vertices) {
	float * result = new float[num_vertices * 2];

	float radius = ((float)length) / 2;

	float theta = 2 * M_PI / (float)num_vertices;

	for (int i = 0; i < num_vertices; i++) {
		float angle = theta * i;
		float y = radius - cosf(angle) * radius;
		float x = radius + sinf(angle) * radius;
		result[2 * i] = x;
		result[2 * i + 1] = y;
	}

	return result;

}

struct int_point make_point(int num_vertices, float * vertices, float center, int num_iterations, std::default_random_engine  &r) {
	float current_x = center;
	float current_y = center;

	std::uniform_int_distribution<int> distribution(0, num_vertices - 1);


	for (int i = 0; i < num_iterations; i++) {
		int vertex = distribution(r);
		current_x += (vertices[vertex * 2] - current_x) / 2;
		current_y += (vertices[vertex * 2 + 1] - current_y) / 2;
	}

	return int_point { (int)current_x, (int)current_y };
		
}

int * count_points(int num_points, int num_vertices, int length, int num_iterations, std::default_random_engine &r) {

	int * counts = new int[length * length]();

	float * vertices = make_vertices(length, num_vertices);

	float center = ((float)length) / 2;


	for (int i = 0; i < num_points; i++) {
		struct int_point p = make_point(num_vertices, vertices, center, num_iterations, r);
		int index = length * p.y + p.x;
		counts[index] += 1;
		if ((i & 33554431) == 0) {
			printf("%f\n", (float)i/(float)num_points);
		}
	}

	delete[] vertices;

	return counts;

}

bool WritePNG(const std::string& filename, int width, int height, const uint16_t* buffer);

void save_fractal(int length, int * counts, char filename[]) {
	int max_count = 0;
	for (int i = 0; i < length * length; i++) {
		max_count = std::max(counts[i], max_count);
	}
	printf("max hit: %i\n", max_count);
	uint16_t * buffer = new uint16_t[length * length];
	for (int i = 0; i < length * length; i++) {
		uint16_t gray = (uint16_t)(65535 * (1 - float(counts[i]) / float(max_count)));
		buffer[i] = gray;
	}

	WritePNG(std::string(filename), length, length, buffer);

	delete[] buffer;

}

bool WritePNG(const std::string& filename, int width, int height, const uint16_t* buffer) {
	png_image img;
	memset(&img, 0, sizeof(img));
	img.version = PNG_IMAGE_VERSION;
	img.width = width;
	img.height = height;
	img.format = PNG_FORMAT_LINEAR_Y;
	img.flags = 0;
	img.colormap_entries = 0;

	// We have to convert from RGB565 to RGB each coded on uint8 because PNG doesn't support RGB565
	const int npixels = width * height;
	

	// negative stride indicates that bottom-most row is first in the buffer (to flip image)
	const int row_stride = width;

	png_image_write_to_file(
		&img, filename.c_str(), false, (void*)buffer, row_stride, NULL
	);

	if (PNG_IMAGE_FAILED(img)) {
		std::cout << "Failed to write image : " << img.message;
		return false;
	}
	else {
		if (img.warning_or_error != 0) {
			std::cout << "libpng warning : " << img.message;
		}
		return true;
	}
}
