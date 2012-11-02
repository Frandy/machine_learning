

CC = g++-4.7

GCCDIR = /usr/local/gcc47/gcc-4.7
CPP_INCLUDE_PATH = $(GCCDIR)/include
CPP_LIBRARY_PATH = $(GCCDIR)/lib

CV_INCLUDE_PATH = /usr/local/include/opencv
CV_LIBRARY_PATH = /usr/local/lib
CV_LIBRARY = -lopencv_core -lopencv_highgui

CFLAGS = -Wall -std=c++0x -I $(CPP_INCLUDE_PATH) -I $(CV_INCLUDE_PATH) 
LFLAGS = -std=c++0x -I $(CPP_INCLUDE_PATH) -I $(CV_INCLUDE_PATH) -L $(CPP_LIBRARY_PATH)  -L $(CV_LIBRARY_PATH)  -lm -lopencv_core -lopencv_highgui

HEADERS = 	../common/commontest.h \
			../kmeans/kmeans.h \
			../knn/knn.h 

INCLUDE_HEADER_PATH = -I ../common/ -I ../kmeans/ -I ../knn/

SOURCES = 	../common/commontest.cpp \
			../kmeans/kmeans.cpp \
			../knn/knn.cpp \
		  	test.cpp

OBJECTS = $(SOURCES:.cpp=.o)

.cpp.o:
	$(CC) -c $(CFLAGS) $(INCLUDE_HEADER_PATH) -o $@ $<

TARGET = test

all:$(TARGET)

$(TARGET):$(OBJECTS) $(HEADERS)
	$(CC) -o $@ $^ $(LFLAGS)

clean:
	rm -Rf $(OBJECTS)	