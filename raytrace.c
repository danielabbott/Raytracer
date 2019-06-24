/*
Camera 2 units wide, centred at (0,0,-1).
Rays originate from (0,0,0) and pass through camera plane ^
Scene is rendered in chunks. For every pixel in a chunk the ray is tested for intersections against
every object in the scene. Each intersection is recorded. Another pass is done on the chunk which takes
that intersection data and applies lighting.
*/

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <stdbool.h>
#include "raytrace.h"

typedef struct RAYHITINFO
{
	// colour of object at position the ray intersected. 
	uint32_t colour; // if alpha = 0, the ray did not hit an object
	float position[3]; // position ray hit the object (world space)
	uint32_t pixelX;
	uint32_t pixelY;
	uint16_t shape;
	uint16_t shapeIndex;
} RAYHITINFO;

#define CHUNK_SIZE 32	

#define SHAPE_SPHERE 1
#define SHAPE_PLANE  2

#define NONE 0
#define SSE 1
#define AVX 2

#ifndef SIMD
#if defined(_M_X64) || defined(_M_AMD64) || defined(__amd64__) || defined(__amd64__) || defined(__x86_64__)  || defined(_x86_64)
#define SIMD SSE
#else
#define SIMD NONE
#endif
#endif

#if SIMD == SSE
#include <xmmintrin.h>
#endif

#if SIMD == AVX
#include <immintrin.h>
#endif

#if SIMD == SSE
#define ELEMENTS_PER_VECTOR 4
#define VEC_T_SIZE 16
#define vec_t __m128
#define COMPARISON_TYPE float

#define mul_ps _mm_mul_ps
#define add_ps _mm_add_ps
#define sub_ps _mm_sub_ps
#define store_ps _mm_store_ps
#define div_ps _mm_div_ps
#define sqrt_ps _mm_sqrt_ps
#define rsqrt_ps _mm_rsqrt_ps
#define set1_ps _mm_set1_ps
#define xor_ps _mm_xor_ps
#define cmpge_ps _mm_cmpge_ps
#define cmpgt_ps _mm_cmpgt_ps
#define cmple_ps _mm_cmple_ps
#define cmplt_ps _mm_cmplt_ps
#define load_ps _mm_load_ps
#define setzero_ps _mm_setzero_ps
#define max_ps _mm_max_ps
#define min_ps _mm_min_ps
#define rcp_ps _mm_rcp_ps
#define get1_ss _mm_cvtss_f32

#elif SIMD == AVX
#define ELEMENTS_PER_VECTOR 8
#define VEC_T_SIZE 32
#define vec_t __m256
#define COMPARISON_TYPE float

#define mul_ps _mm256_mul_ps
#define add_ps _mm256_add_ps
#define sub_ps _mm256_sub_ps
#define store_ps _mm256_store_ps
#define div_ps _mm256_div_ps
#define sqrt_ps _mm256_sqrt_ps
#define rsqrt_ps _mm256_rsqrt_ps
#define set1_ps _mm256_set1_ps
#define xor_ps _mm256_xor_ps
#define load_ps _mm256_load_ps
#define setzero_ps _mm256_setzero_ps
#define max_ps _mm256_max_ps
#define min_ps _mm256_min_ps
#define rcp_ps _mm256_rcp_ps
#define get1_ss _mm256_cvtss_f32


#else

#define ELEMENTS_PER_VECTOR 1
#define VEC_T_SIZE 4
#define vec_t float
#define COMPARISON_TYPE uint32_t

static inline vec_t mul_ps(vec_t a, vec_t b) { return a * b; }
static inline vec_t add_ps(vec_t a, vec_t b) { return a + b; }
static inline vec_t sub_ps(vec_t a, vec_t b) { return a - b; }
static inline void store_ps(vec_t * ptr, vec_t a) { *ptr = a; }
static inline vec_t div_ps(vec_t a, vec_t b) { return a / b; }
static inline vec_t sqrt_ps(vec_t a) { return (vec_t)sqrt(a); }
static inline vec_t rsqrt_ps(vec_t a) { return 1.0f / (vec_t)sqrt(a); }
static inline vec_t set1_ps(vec_t a) { return a; }
static inline vec_t load_ps(vec_t* a) { return *a; }
static inline vec_t setzero_ps() { return 0.0f; }
static inline vec_t max_ps(vec_t a, vec_t b) { return a > b ? a : b; }
static inline vec_t min_ps(vec_t a, vec_t b) { return a > b ? b : a; }
static inline vec_t rcp_ps(vec_t a) { return 1.0f / a; }
static inline vec_t get1_ss(vec_t a) { return a; }
static inline vec_t absolute_vector_components(vec_t a) { return a >= 0 ? a : -a; }

#endif

#if SIMD == SSE || SIMD == AVX
vec_t absolute_vector_components(vec_t a)
{
	// Clears sign bit
	return xor_ps(a, set1_ps(-0.0f));
}
#endif

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#define ALIGN_PREFIX __declspec(align(VEC_T_SIZE))
#define ALIGN_SUFFIX
#else
#define ALIGN_PREFIX
#define ALIGN_SUFFIX __attribute__((align(VEC_T_SIZE)))
#endif

#ifdef DEBUG
static inline void assert(int condition)
{
	if (!condition) {
		printf("ASSERTATION FAIL\n");
		abort();
	}
}
#else
static inline void assert(int condition) { (void)condition; }
#endif

static inline vec_t dot(vec_t ax, vec_t ay, vec_t az, vec_t bx, vec_t by, vec_t bz)
{
	return add_ps(add_ps(mul_ps(ax, bx),
		mul_ps(ay, by)),
		mul_ps(az, bz));
}

void raytracer_create(RAYTRACER * rt)
{
	if (!rt->imageWidth || !rt->imageHeight || (rt->imageWidth % ELEMENTS_PER_VECTOR))
		return;

	if (!rt->pixels) {
		rt->pixels = malloc(rt->imageWidth * rt->imageHeight * 4);
		for (unsigned int i = 0; i < rt->imageHeight * rt->imageHeight; i++)
			rt->pixels[i] = 0xff000000;
	}

	/* Pre-calculate radius squared for spheres */

	for (unsigned int i = 0; i < rt->spheresCount; i++) {
		rt->spheres[i].radiusSquared = rt->spheres[i].radius * rt->spheres[i].radius;
	}

	/* Check for and correct invalid values in point light structs */

	for (unsigned int i = 0; i < rt->pointLightsCount; i++) {
		if (rt->pointLights[i].attenuation <= 0)
			rt->pointLights[i].attenuation = 0.001f;
		if (!rt->pointLights[i].colour) {
			rt->pointLights[i].colour = 0xff010101;
		}
	}
}

/* Returns the first intersection with a sphere for each ray */
/* nearestIntersections,intersectionMaterials,intersectionShapeIndex should be arrays of length ELEMENTS_PER_VECTOR */
/* nearestIntersections, intersectionMaterials, intersectionShapeIndex are the output */
/* ignore should point to an unsigned int array of length ELEMENTS_PER_VECTOR. To specifiy that no sphere should be ignored, set ignore[x] to -1 */
static void do_sphere_intersection(RAYTRACER * rt,
	float * nearestIntersections, MATERIAL * intersectionMaterials, int * intersectionShapeIndex,
	vec_t rayOriginX, vec_t rayOriginY, vec_t rayOriginZ,
	vec_t rayDirectionX, vec_t rayDirectionY, vec_t rayDirectionZ, int * ignore)
{
	for (unsigned int i = 0; i < rt->spheresCount; i++) {
		// ray-sphere intersection

		// a = D^2
		// b = 2OD
		// c = O^2 - R^2
		// where D = ray direction, O = ray origin and R = sphere radius

		// dot product of rayDirection with itself
		vec_t a = dot(rayDirectionX, rayDirectionY, rayDirectionZ, rayDirectionX, rayDirectionY, rayDirectionZ);

		vec_t Ox = sub_ps(rayOriginX, set1_ps(rt->spheres[i].position[0]));
		vec_t Oy = sub_ps(rayOriginY, set1_ps(rt->spheres[i].position[1]));
		vec_t Oz = sub_ps(rayOriginZ, set1_ps(rt->spheres[i].position[2]));

		vec_t b = mul_ps(dot(rayDirectionX, rayDirectionY, rayDirectionZ, Ox, Oy, Oz), set1_ps(2));

		vec_t c = sub_ps(dot(Ox, Oy, Oz, Ox, Oy, Oz), set1_ps(rt->spheres[i].radiusSquared));

		// b^2 - 4ac
		vec_t discriminant = sub_ps(mul_ps(b, b), mul_ps(set1_ps(4), mul_ps(a, c)));

		/* If the discriminant is negative then there are no solutions */

#if SIMD != NONE
		ALIGN_PREFIX COMPARISON_TYPE comparison[ELEMENTS_PER_VECTOR] ALIGN_SUFFIX;
#endif

		int atLeastOneRayHasASolution = false;

		// compare discriminants to 0
#if SIMD == SSE
		store_ps(comparison, cmpge_ps(discriminant, setzero_ps()));
#elif SIMD == AVX
		store_ps(comparison, _mm256_cmp_ps(discriminant, setzero_ps(), _CMP_GE_OS));
#elif SIMD == NONE
		atLeastOneRayHasASolution = discriminant >= 0;
#endif

#if SIMD != NONE
		for (unsigned int j = 0; j < ELEMENTS_PER_VECTOR; j++) {
			if (comparison[j] != 0) {
				atLeastOneRayHasASolution = true;
				break;
			}
		}
#endif

		/* The calculation only procedes if at least one of the rays being tested has intersected */

		if (atLeastOneRayHasASolution) {
			/* Use quadratic forumula to find solution */

			vec_t sqrtDiscrim = sqrt_ps(discriminant);

			vec_t negB = absolute_vector_components(b);

			ALIGN_PREFIX float solution1[ELEMENTS_PER_VECTOR] ALIGN_SUFFIX;
			ALIGN_PREFIX float solution2[ELEMENTS_PER_VECTOR] ALIGN_SUFFIX;

			vec_t denominator = add_ps(a, a);
			store_ps(solution1, div_ps(add_ps(negB, sqrtDiscrim), denominator));
			store_ps(solution2, div_ps(sub_ps(negB, sqrtDiscrim), denominator));

			for (unsigned int j = 0; j < ELEMENTS_PER_VECTOR; j++) {
				if (ignore[j] != i) {
#if SIMD != NONE
					if (comparison[j]) {
#endif
						/* Store largest solution */

						float solution = solution1[j];

						if (solution < 0 || (solution2[j] > 0 && solution2[j] < solution))
							solution = solution2[j];

						if (solution > 0 && solution < nearestIntersections[j]) {
							nearestIntersections[j] = solution;
							intersectionMaterials[j].colour = rt->spheres[i].colour;
							intersectionShapeIndex[j] = i;
						}
#if SIMD != NONE
					}
#endif
				}
			}
		}

	}
}

/* Returns the first intersection with a plane for each ray */
/* intersections,intersectionMaterials,intersectionShapeIndex should be arrays of length ELEMENTS_PER_VECTOR */
/* nearestIntersections, intersectionMaterials, intersectionShapeIndex are the output */
/* ignore should point to an unsigned int array of length ELEMENTS_PER_VECTOR. To specifiy that no plane should be ignored, set ignore[x] to -1 */
static void do_plane_intersection(RAYTRACER * rt,
	float * intersections, MATERIAL * intersectionMaterials, int * intersectionShapeIndex,
	vec_t rayOriginX, vec_t rayOriginY, vec_t rayOriginZ,
	vec_t rayDirectionX, vec_t rayDirectionY, vec_t rayDirectionZ, int * ignore)
{
	for (unsigned int i = 0; i < rt->planesCount; i++) {
		// t = ((p0) * n) / (l * n)
		// where p0 = centre position of plane - ray origin
		// 		 n = plane normal
		//		 l = ray direction

		vec_t planeNormalX = set1_ps(rt->planes[i].normal[0]);
		vec_t planeNormalY = set1_ps(rt->planes[i].normal[1]);
		vec_t planeNormalZ = set1_ps(rt->planes[i].normal[2]);

		vec_t denominator = dot(rayDirectionX, rayDirectionY, rayDirectionZ, planeNormalX, planeNormalY, planeNormalZ);

		// denominator < 0 ?

#if SIMD != NONE
		ALIGN_PREFIX COMPARISON_TYPE comparison[ELEMENTS_PER_VECTOR] ALIGN_SUFFIX;
#endif

		int atLeastOneRayHasASolution = 0;

#if SIMD == SSE
		store_ps(comparison, cmple_ps(denominator, setzero_ps()));
#elif SIMD == AVX
		store_ps(comparison, _mm256_cmp_ps(denominator, setzero_ps(), _CMP_LE_OS));
#elif SIMD == NONE
		atLeastOneRayHasASolution = denominator < 0;
#endif

#if SIMD != NONE
		for (unsigned int j = 0; j < ELEMENTS_PER_VECTOR; j++) {
			if (((uint32_t *)comparison)[j]) {
				atLeastOneRayHasASolution = 1;
				break;
			}
		}
#endif

		if (atLeastOneRayHasASolution) {

			vec_t planePositionX = set1_ps(rt->planes[i].position[0]);
			vec_t planePositionY = set1_ps(rt->planes[i].position[1]);
			vec_t planePositionZ = set1_ps(rt->planes[i].position[2]);

			vec_t px = sub_ps(planePositionX, rayOriginX);
			vec_t py = sub_ps(planePositionY, rayOriginY);
			vec_t pz = sub_ps(planePositionZ, rayOriginZ);

			vec_t numerator = dot(px, py, pz, planeNormalX, planeNormalY, planeNormalZ);

			vec_t t = div_ps(numerator, denominator);

			ALIGN_PREFIX float solution[ELEMENTS_PER_VECTOR] ALIGN_SUFFIX;

			store_ps(solution, t);

			for (unsigned int j = 0; j < ELEMENTS_PER_VECTOR; j++) {
				if (ignore[j] != i) {
#if SIMD != NONE
					if (comparison[j]) {
#endif
						if (solution[j] > 0 && solution[j] < intersections[j]) {
							intersections[j] = solution[j];
							intersectionMaterials[j].colour = rt->planes[i].colour;
							intersectionShapeIndex[j] = i;
						}
#if SIMD != NONE
					}
#endif
				}
			}

		}
	}
}

void apply_point_lights(RAYTRACER * rt, RAYHITINFO * rayHitInfo, unsigned int rayCount)
{
	assert(rayCount <= ELEMENTS_PER_VECTOR && rayCount);

	/* Find surface normal */

	ALIGN_PREFIX float x_floats[ELEMENTS_PER_VECTOR] ALIGN_SUFFIX = { 0 };
	ALIGN_PREFIX float y_floats[ELEMENTS_PER_VECTOR] ALIGN_SUFFIX = { 0 };
	ALIGN_PREFIX float z_floats[ELEMENTS_PER_VECTOR] ALIGN_SUFFIX = { 0 };


	for (unsigned int i = 0; i < rayCount; i++) {
		if (rayHitInfo[i].shape == SHAPE_SPHERE) { /* Sphere */
			// Normal = norm(intersect - origin)

			float px = rayHitInfo[i].position[0];
			float py = rayHitInfo[i].position[1];
			float pz = rayHitInfo[i].position[2];

			px -= rt->spheres[rayHitInfo[i].shapeIndex].position[0];
			py -= rt->spheres[rayHitInfo[i].shapeIndex].position[1];
			pz -= rt->spheres[rayHitInfo[i].shapeIndex].position[2];

			float lenR = 1.0f / (float)sqrt(px*px + py*py + pz*pz);

			x_floats[i] = px * lenR;
			y_floats[i] = py * lenR;
			z_floats[i] = pz * lenR;
		}
		else if (rayHitInfo[i].shape == SHAPE_PLANE) { /* Plane */
			x_floats[i] = rt->planes[rayHitInfo[i].shapeIndex].normal[0];
			y_floats[i] = rt->planes[rayHitInfo[i].shapeIndex].normal[1];
			z_floats[i] = rt->planes[rayHitInfo[i].shapeIndex].normal[2];
		}
		else {
			assert(0);
		}
	}

	vec_t objectSuraceNormalX = load_ps(x_floats);
	vec_t objectSuraceNormalY = load_ps(y_floats);
	vec_t objectSuraceNormalZ = load_ps(z_floats);

	/* Load ray intersection points into vector types */

	for (unsigned int i = 0; i < rayCount; i++) {
		x_floats[i] = rayHitInfo[i].position[0];
		y_floats[i] = rayHitInfo[i].position[1];
		z_floats[i] = rayHitInfo[i].position[2];
	}
	vec_t intersectX = load_ps(x_floats);
	vec_t intersectY = load_ps(y_floats);
	vec_t intersectZ = load_ps(z_floats);

	float r[ELEMENTS_PER_VECTOR] = { 0 };
	float g[ELEMENTS_PER_VECTOR] = { 0 };
	float b[ELEMENTS_PER_VECTOR] = { 0 };

	for (unsigned int j = 0; j < rayCount; j++) {
		r[j] = ((rayHitInfo[j].colour & 0xff0000) >> 16) / 255.0f;
		g[j] = ((rayHitInfo[j].colour & 0xff00) >> 8) / 255.0f;
		b[j] = (rayHitInfo[j].colour & 0xff) / 255.0f;
	}

	ALIGN_PREFIX float newRed[ELEMENTS_PER_VECTOR] ALIGN_SUFFIX;
	ALIGN_PREFIX float newGreen[ELEMENTS_PER_VECTOR] ALIGN_SUFFIX;
	ALIGN_PREFIX float newBlue[ELEMENTS_PER_VECTOR] ALIGN_SUFFIX;

	memset(newRed, 0, ELEMENTS_PER_VECTOR * sizeof(float));
	memset(newGreen, 0, ELEMENTS_PER_VECTOR * sizeof(float));
	memset(newBlue, 0, ELEMENTS_PER_VECTOR * sizeof(float));

	for (unsigned int i = 0; i < rt->pointLightsCount; i++) {
		/* Get light position */

		vec_t lightX = set1_ps(rt->pointLights[i].position[0]);
		vec_t lightY = set1_ps(rt->pointLights[i].position[1]);
		vec_t lightZ = set1_ps(rt->pointLights[i].position[2]);

		/* Subtract ray intersect position to get object->light vector*/

		vec_t objectToLightX = sub_ps(lightX, intersectX);
		vec_t objectToLightY = sub_ps(lightY, intersectY);
		vec_t objectToLightZ = sub_ps(lightZ, intersectZ);

		/* Normalise */

		vec_t lightRayLength = sqrt_ps(add_ps(add_ps(mul_ps(objectToLightX, objectToLightX),
			mul_ps(objectToLightY, objectToLightY)),
			mul_ps(objectToLightZ, objectToLightZ)));

		vec_t multiplier = rcp_ps(lightRayLength);

		vec_t objectToLightXNormalised = mul_ps(objectToLightX, multiplier);
		vec_t objectToLightYNormalised = mul_ps(objectToLightY, multiplier);
		vec_t objectToLightZNormalised = mul_ps(objectToLightZ, multiplier);

#ifndef DISABLE_SHADOWS
		
		/* Check for objects casting a shadow */


		ALIGN_PREFIX float nearestIntersections[ELEMENTS_PER_VECTOR] ALIGN_SUFFIX;
		MATERIAL mat[ELEMENTS_PER_VECTOR];

		for (unsigned int j = 0; j < ELEMENTS_PER_VECTOR; j++)
			nearestIntersections[j] = INFINITY;

		int intersectionShapeIndex[ELEMENTS_PER_VECTOR]; // index into spheres, planes, etc. arrays

		/* Create an array to specify whether each ray should ignore any specific sphere or not */

		unsigned int ignore[ELEMENTS_PER_VECTOR] = { 0 };
		for (unsigned int j = 0; j < ELEMENTS_PER_VECTOR; j++) {
			if (rayHitInfo[j].shape == SHAPE_SPHERE) {
				ignore[j] = rayHitInfo[j].shapeIndex;
			}
			else {
				ignore[j] = (unsigned int)-1;
			}
		}

		do_sphere_intersection(rt, nearestIntersections, mat, intersectionShapeIndex,
			intersectX, intersectY, intersectZ,
			objectToLightXNormalised, objectToLightYNormalised, objectToLightZNormalised, ignore);

		/* Create an array to specify whether each ray should ignore any specific plane or not */

		memset(ignore, 0, ELEMENTS_PER_VECTOR * sizeof(unsigned int));
		for (unsigned int j = 0; j < ELEMENTS_PER_VECTOR; j++) {
			if (rayHitInfo[j].shape == SHAPE_PLANE) {
				ignore[j] = rayHitInfo[j].shapeIndex;
			}
			else {
				ignore[j] = (unsigned int)-1;
			}
		}

		do_plane_intersection(rt, nearestIntersections, mat, intersectionShapeIndex,
			intersectX, intersectY, intersectZ,
			objectToLightXNormalised, objectToLightYNormalised, objectToLightZNormalised, ignore);


		/* Compare the distance to the light to the distance to the object */

		// both must be true for the object to be casting a shadow
		ALIGN_PREFIX COMPARISON_TYPE lightRayIntersectionComparison[ELEMENTS_PER_VECTOR] ALIGN_SUFFIX;
		ALIGN_PREFIX COMPARISON_TYPE lightRayIntersectionComparison2[ELEMENTS_PER_VECTOR] ALIGN_SUFFIX;

#if SIMD == SSE
		vec_t n = load_ps(nearestIntersections);
		store_ps(lightRayIntersectionComparison, cmplt_ps(n, lightRayLength));
		store_ps(lightRayIntersectionComparison2, cmpge_ps(n, set1_ps(0.01f)));
#elif SIMD == AVX
		vec_t n = load_ps(nearestIntersections);
		store_ps(lightRayIntersectionComparison, _mm256_cmp_ps(n, lightRayLength, _CMP_LT_OS));
		store_ps(lightRayIntersectionComparison2, _mm256_cmp_ps(n, set1_ps(0.01f), _CMP_GE_OS));
#elif SIMD == NONE
		lightRayIntersectionComparison[0] = nearestIntersections[0] < lightRayLength;
		lightRayIntersectionComparison2[0] = nearestIntersections[0] >= 0.01f;
#endif

#endif

		/* Use surface normal to calculate intensity */

		// dot(objectSuraceNormal, objectToLight)

		vec_t intensity = dot(objectSuraceNormalX, objectSuraceNormalY, objectSuraceNormalZ, 
			objectToLightXNormalised, objectToLightYNormalised, objectToLightZNormalised);

		ALIGN_PREFIX float floats[ELEMENTS_PER_VECTOR] ALIGN_SUFFIX;

#ifndef DISABLE_SHADOWS

		/* Set intensity to 0 if a shadow is cast */

		store_ps(floats, intensity);

		for (unsigned int j = 0; j < rayCount; j++) {
			if (((uint32_t *)lightRayIntersectionComparison)[j] && ((uint32_t *)lightRayIntersectionComparison2)[j]) {
				floats[j] = 0;
			}
		}

		intensity = load_ps(floats);

#endif


		// intensity = clamp(intensity, 0, 1)
		intensity = min_ps(max_ps(setzero_ps(), intensity), set1_ps(1));

		/* Get length of object->light vector */

		vec_t distanceToLight = lightRayLength;

		distanceToLight = max_ps(set1_ps(0.00001f), distanceToLight);


		/* Apply inverse-square law to find light intensity */


		intensity = div_ps(intensity, mul_ps(mul_ps(distanceToLight, distanceToLight), set1_ps(rt->pointLights[i].attenuation)));

		store_ps(floats, intensity);

		for (unsigned int j = 0; j < rayCount; j++) {
			float a = floats[j];
			if (!isnormal(a))
				continue;

			newRed[j] += r[j] * a * (((rt->pointLights[i].colour & 0xff0000) >> 16) / 255.0f);
			newGreen[j] += g[j] * a * (((rt->pointLights[i].colour & 0xff00) >> 8) / 255.0f);
			newBlue[j] += b[j] * a * ((rt->pointLights[i].colour & 0xff) / 255.0f);
		}

	}

	store_ps(newRed, min_ps(max_ps(load_ps(newRed), setzero_ps()), set1_ps(1)));
	store_ps(newGreen, min_ps(max_ps(load_ps(newGreen), setzero_ps()), set1_ps(1)));
	store_ps(newBlue, min_ps(max_ps(load_ps(newBlue), setzero_ps()), set1_ps(1)));

	for (unsigned int j = 0; j < rayCount; j++) {
		rayHitInfo[j].colour = 0xff000000 | (((uint8_t)(newRed[j] * 255.0f)) << 16) | (((uint8_t)(newGreen[j] * 255.0f)) << 8) | ((uint8_t)(newBlue[j] * 255.0f));
	}
}

static inline void raytracer_render_(RAYTRACER * rt, float pixelXCamSpaceOffset, float pixelYCamSpaceOffset,
	RAYHITINFO * rayHitInfo, unsigned int x, unsigned int y, unsigned int w, unsigned int h)
{
	assert(w && h && pixelXCamSpaceOffset > 0 && pixelYCamSpaceOffset > 0);

	unsigned int rayHitInfoCount = 0;

	/* Cast initial rays */

	float aspect = rt->imageWidth / (float)rt->imageHeight;
	float pixelYCamSpace = 1 / aspect - pixelYCamSpaceOffset * y;

	ALIGN_PREFIX float pixelCamSpaceX[ELEMENTS_PER_VECTOR] ALIGN_SUFFIX;

	for (unsigned int pixelY = 0; pixelY < h; pixelY++) {

		for (unsigned int pixelX = 0; pixelX < w; pixelX += ELEMENTS_PER_VECTOR) {


			pixelCamSpaceX[0] = -1.0f + pixelXCamSpaceOffset * (x + pixelX);
#if ELEMENTS_PER_VECTOR > 1
			pixelCamSpaceX[1] = -1.0f + pixelXCamSpaceOffset * (x + pixelX + 1);
			pixelCamSpaceX[2] = -1.0f + pixelXCamSpaceOffset * (x + pixelX + 2);
			pixelCamSpaceX[3] = -1.0f + pixelXCamSpaceOffset * (x + pixelX + 3);
#if ELEMENTS_PER_VECTOR > 4
			pixelCamSpaceX[4] = -1.0f + pixelXCamSpaceOffset * (x + pixelX + 4);
			pixelCamSpaceX[5] = -1.0f + pixelXCamSpaceOffset * (x + pixelX + 5);
			pixelCamSpaceX[6] = -1.0f + pixelXCamSpaceOffset * (x + pixelX + 6);
			pixelCamSpaceX[7] = -1.0f + pixelXCamSpaceOffset * (x + pixelX + 7);
#endif
#endif

			vec_t rayDirectionX = load_ps(pixelCamSpaceX);
			vec_t rayDirectionY = set1_ps(pixelYCamSpace);
			float neg1 = -1;
			vec_t rayDirectionZ = set1_ps(neg1);

			/* Normalise direction vector */

			vec_t rayDirectionMultiplier = rsqrt_ps(add_ps(add_ps(mul_ps(rayDirectionX, rayDirectionX),
				mul_ps(rayDirectionY, rayDirectionY)),
				mul_ps(rayDirectionZ, rayDirectionZ)));

			rayDirectionX = mul_ps(rayDirectionX, rayDirectionMultiplier);
			rayDirectionY = mul_ps(rayDirectionY, rayDirectionMultiplier);
			rayDirectionZ = mul_ps(rayDirectionZ, rayDirectionMultiplier);

			/* Check for intersections between ray and all objects in the scene */

			/* Check for intersections with spheres */

			float nearestIntersections[ELEMENTS_PER_VECTOR];

			for (unsigned int i = 0; i < ELEMENTS_PER_VECTOR; i++)
				nearestIntersections[i] = INFINITY;

			MATERIAL intersectionMaterials[ELEMENTS_PER_VECTOR] = { 0 };
			int intersectionShapeIndex[ELEMENTS_PER_VECTOR] = { 0 }; // index into spheres, planes, etc.
			int intersectionType[ELEMENTS_PER_VECTOR] = { 0 };

			const vec_t zero = setzero_ps();

			int ignore[ELEMENTS_PER_VECTOR];
			memset(ignore, 0xff, ELEMENTS_PER_VECTOR * sizeof(unsigned int));

			do_sphere_intersection(rt, nearestIntersections, intersectionMaterials, intersectionShapeIndex,
				zero, zero, zero,
				rayDirectionX, rayDirectionY, rayDirectionZ, ignore);

			for (unsigned int i = 0; i < ELEMENTS_PER_VECTOR; i++) {
				if (nearestIntersections[i] < INFINITY)
					intersectionType[i] = SHAPE_SPHERE;
			}

			do_plane_intersection(rt, nearestIntersections, intersectionMaterials, intersectionShapeIndex,
				zero, zero, zero,
				rayDirectionX, rayDirectionY, rayDirectionZ, ignore);

			for (unsigned int i = 0; i < ELEMENTS_PER_VECTOR; i++) {
				if (nearestIntersections[i] < INFINITY && !intersectionType[i])
					intersectionType[i] = SHAPE_PLANE;
			}

			/* Store the intersection data for batch processing */

			ALIGN_PREFIX float rayDirectionX_floats[ELEMENTS_PER_VECTOR] ALIGN_SUFFIX;
			store_ps(rayDirectionX_floats, rayDirectionX);

			ALIGN_PREFIX float rayDirectionY_floats[ELEMENTS_PER_VECTOR] ALIGN_SUFFIX;
			store_ps(rayDirectionY_floats, rayDirectionY);

			ALIGN_PREFIX float rayDirectionZ_floats[ELEMENTS_PER_VECTOR] ALIGN_SUFFIX;
			store_ps(rayDirectionZ_floats, rayDirectionZ);

			for (int i = 0; i < ELEMENTS_PER_VECTOR; i++) {
				if (nearestIntersections[i] < INFINITY) {
					rayHitInfo[rayHitInfoCount].pixelX = x + pixelX + i;
					rayHitInfo[rayHitInfoCount].pixelY = y + pixelY;

					rayHitInfo[rayHitInfoCount].shape = (uint16_t)intersectionType[i];
					rayHitInfo[rayHitInfoCount].shapeIndex = (uint16_t)intersectionShapeIndex[i];


					// direction * t = point of intersection in 3d space
					float px = rayDirectionX_floats[i] * nearestIntersections[i];
					float py = rayDirectionY_floats[i] * nearestIntersections[i];
					float pz = rayDirectionZ_floats[i] * nearestIntersections[i];

					rayHitInfo[rayHitInfoCount].position[0] = px;
					rayHitInfo[rayHitInfoCount].position[1] = py;
					rayHitInfo[rayHitInfoCount].position[2] = pz;

					int black = 0;

					if (rayHitInfo[rayHitInfoCount].shape == SHAPE_PLANE && rt->planes[rayHitInfo[rayHitInfoCount].shapeIndex].chequered > 0) {
						/* Apply chequered pattern to plane */

						float c = rt->planes[rayHitInfo[rayHitInfoCount].shapeIndex].chequered;

						int x = (int)round(px / c);
						int z = (int)round(pz / c);

						if (x < 0) x -= (int)c;

						if ((abs(x) % 2) ^ (abs(z) % 2)) {
							black = 1;
						}
					}

					rayHitInfo[rayHitInfoCount].colour = black ? 0xff000000 : intersectionMaterials[i].colour;

					rayHitInfoCount++;
				}
			}
		}
		pixelYCamSpace -= pixelYCamSpaceOffset;
	}


	/* Apply point lights in batches */
	/* (For better SIMD utilisation - rays with that intersected are grouped together) */

	unsigned int rayHitsRemaining = rayHitInfoCount;

	while (rayHitsRemaining >= ELEMENTS_PER_VECTOR) {
		apply_point_lights(rt, rayHitInfo, ELEMENTS_PER_VECTOR);

		for (unsigned int j = 0; j < ELEMENTS_PER_VECTOR; j++) {
			unsigned int px = rayHitInfo[j].pixelX;
			unsigned int py = rayHitInfo[j].pixelY;
			assert(py * rt->pixelsStride + px < rt->imageWidth * rt->imageHeight);
			rt->pixels[py * rt->pixelsStride + px] = rayHitInfo[j].colour;
		}

		rayHitInfo += ELEMENTS_PER_VECTOR;
		rayHitsRemaining -= ELEMENTS_PER_VECTOR;
	}

#if SIMD != NONE
	if (rayHitsRemaining) {
		apply_point_lights(rt, rayHitInfo, rayHitsRemaining);

		for (unsigned int j = 0; j < rayHitsRemaining; j++) {
			unsigned int px = rayHitInfo[j].pixelX;
			unsigned int py = rayHitInfo[j].pixelY;
			assert(py * rt->pixelsStride + px < rt->imageWidth * rt->imageHeight);
			rt->pixels[py * rt->pixelsStride + px] = rayHitInfo[j].colour;
		}
	}
#endif

}

void raytracer_render(RAYTRACER * rt)
{
	float pixelXCamSpaceOffset = 2 / (float)rt->imageWidth;

	float pixelYCamSpaceOffset = (2.0f / ((float)rt->imageWidth / (float)rt->imageHeight)) / (float)rt->imageHeight;

	RAYHITINFO * rayHitInfo = malloc(CHUNK_SIZE * CHUNK_SIZE * sizeof(RAYHITINFO));

	int rowsToGo = rt->imageHeight;
	unsigned int y = 0;

	while (rowsToGo > 0) {
		unsigned int areaHeight = rowsToGo > CHUNK_SIZE ? CHUNK_SIZE : rowsToGo;
		unsigned int x = 0;
		int columnsToGo = rt->imageWidth;

		while (columnsToGo > 0) {
			unsigned int areaWidth = columnsToGo > CHUNK_SIZE ? CHUNK_SIZE : columnsToGo;

			raytracer_render_(rt, pixelXCamSpaceOffset, pixelYCamSpaceOffset, rayHitInfo, x, y, areaWidth, areaHeight);
			x += areaWidth;
			columnsToGo -= areaWidth;
		}
		y += areaHeight;
		rowsToGo -= areaHeight;
	}

	free(rayHitInfo);
}


void free_raytracer(RAYTRACER * rt)
{
	free(rt->pixels);
}