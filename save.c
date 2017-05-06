#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#pragma pack(push, 1)
#endif

struct BitmapHeader
{
	uint16_t signature; // 'BM' (0x4D42 little endian)
	uint32_t fileSize;
	uint32_t reserved;
	uint32_t imageDataOffset;

	// DIB Header

	uint32_t dibHeaderSize; // 40
	int32_t imageWidth;
	int32_t imageHeight;
	uint16_t colourPlanes; // 1
	uint16_t bitsPerPixel; // 32
	uint32_t compression; // 0
	uint32_t imageSize; // imageWidth * imageHeight * 4
	int32_t resolutionX; // pixels per meter
	int32_t resolutionY;
	uint32_t coloursInPallet; // 0
	uint32_t importantColours; // 0
}
#if !defined(WIN32) && !defined(_WIN32) && !defined(__WIN32)
__attribute__((packed))
#endif
;

#if defined(WIN32) || defined(_WIN32) || defined(__WIN32)
#pragma pack(pop)
#endif

/* Saves an image as a .BMP file. Only 32-bpp images are supported. */
void save_bitmap_image(uint8_t * pixels, unsigned int width, unsigned int height, const char * filepath)
{
	struct BitmapHeader bmp;
	memset(&bmp, 0, sizeof bmp);
	bmp.signature = 0x4D42;
	bmp.fileSize = sizeof bmp + width*height * 3;
	bmp.imageDataOffset = sizeof bmp;
	bmp.dibHeaderSize = 40;
	bmp.imageWidth = width;
	bmp.imageHeight = height;
	bmp.colourPlanes = 1;
	bmp.bitsPerPixel = 24;
	bmp.compression = 0;
	bmp.imageSize = width * height * 3;
	bmp.resolutionX = 10000;
	bmp.resolutionY = (int)((height / (float)width) * 10000);
	bmp.coloursInPallet = 0;
	bmp.importantColours = 0;

	FILE * file = fopen(filepath, "wb");
	if (!file) {
		fprintf(stderr, "ERROR WRITING BITMAP TO FILE %s\n", filepath);
		return;
	}
	fwrite(&bmp, sizeof bmp, 1, file);

	uint8_t * data = malloc(width * height * 3);

	unsigned int src = 0;
	for (unsigned int y = 0; y < height; y++) {
		unsigned dst = (height - y - 1)*width * 3;

		for (unsigned int x = 0; x < width; x++) {
			data[dst] = pixels[src];
			data[dst + 1] = pixels[src + 1];
			data[dst + 2] = pixels[src + 2];

			src += 4;
			dst += 3;
		}
	}

	fwrite(data, width * height * 3, 1, file);
	
	fclose(file);
	free(data);

}