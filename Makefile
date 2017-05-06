SIMD = 2

SOURCES=raytrace.c main.c save.c
OBJECTS=$(SOURCES:.c=.o)
CFLAGS=-c -Wall -Wextra -O3 -Wno-attributes -Wpedantic -DSIMD=$(SIMD)
LDFLAGS =  -lm -O3

ifeq ($(SIMD), 1)
CFLAGS += -msse
LDFLAGS += -msse
else ifeq ($(SIMD), 2)
CFLAGS += -mavx
LDFLAGS += -mavx
endif

all: $(OBJECTS)
	$(CC) *.o -o raytracer $(LDFLAGS)

.c.o:
	$(CC) $(CFLAGS) $< -o $@

