#ifndef _RAYTRACER_H_
#define _RAYTRACER_H_

// ! Pixel format is ARGB where A (Alpha) is the most signifiant byte

typedef struct SPHERE
{
	float radius;
	float radiusSquared;
	float position[3];
	uint32_t colour;
} SPHERE;

typedef struct MATERIAL
{
	uint32_t colour;
} MATERIAL;

typedef struct POINT_LIGHT
{
	uint32_t colour;
	float position[3];
	float attenuation;
} POINT_LIGHT;

typedef struct PLANE
{
	uint32_t colour;
	float position[3];
	float normal[3];
	float chequered; // width of the squares, only looks correct if the normal is (0,1,0) or (0,-1,0)
} PLANE;

typedef struct RAYTRACER
{
	// Set these fields before calling raytracer_create

	unsigned int spheresCount;
	SPHERE * spheres;

	unsigned int pointLightsCount;
	POINT_LIGHT * pointLights;

	unsigned int planesCount;
	PLANE * planes;

	uint32_t * pixels;
	unsigned int pixelsStride; // if in doubt, set to imageWidth
	unsigned int imageWidth; // MUST BE A MULTIPLE OF 4 for SSE, MULTIPLE OF 8 FOR AVX
	unsigned int imageHeight;

} RAYTRACER;

/* RAYTRACER should already have all fields initialised */
void raytracer_create(RAYTRACER *);

void raytracer_render(RAYTRACER *);

void free_raytracer(RAYTRACER *);

#endif
