#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "raytrace.h"

void save_bitmap_image(uint8_t * pixels, unsigned int width, unsigned int height, const char * filepath);

SPHERE sphere[4] = {
{
	.radius = 1.0f,
	.colour = 0xffff00ff,
	.position = { 1.5f,0,-2.6f }
},
{
	.radius = 1.0f,
	.colour = 0xffffff00,
	.position = { -1,0,-2 }
},
{
	.radius = 0.3f,
	.colour = 0x00ffffff,
	.position = { 0.6f,-0.7f,-1.5f }
},
{
	.radius = 1.3f,
	.colour = 0x00ffffff,
	.position = { 0,-0,-7 }
}
};

POINT_LIGHT plight = {
	.colour = 0xffe0e0e0,
	.position = { 0 ,1, 0 },
	.attenuation = 0.4f
};

PLANE plane = {
	.colour = 0xff808080,
	.position = { 0,-1,0 },
	.normal = { 0,1,0 },
	.chequered = 0.25f,
};

int main(void)
{

#define IMAGE_W 1024
#define IMAGE_H 512

	RAYTRACER rt;
	rt.pixels = NULL;
	rt.imageWidth = IMAGE_W;
	rt.imageHeight = IMAGE_H;
	rt.pixelsStride = IMAGE_W;

	rt.spheresCount = sizeof sphere / sizeof(SPHERE);
	rt.spheres = sphere;

	rt.pointLightsCount = 1;
	rt.pointLights = &plight;

	rt.planesCount = 1;
	rt.planes = &plane;

	raytracer_create(&rt);

	clock_t start = clock(), diff;
	raytracer_render(&rt);
	diff = clock() - start;

	int msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("Render took %d seconds %d milliseconds of CPU time\n", msec / 1000, msec % 1000);

	save_bitmap_image((uint8_t *)rt.pixels, IMAGE_W, IMAGE_H, "out.bmp");

	free_raytracer(&rt);

	//getc(stdin);

	return 0;
}