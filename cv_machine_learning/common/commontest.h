/*
 * commontest.h
 *
 *  Created on: Nov 1, 2012
 *      Author: chjd
 */

#ifndef COMMONTEST_H_
#define COMMONTEST_H_

#include <cxcore.h>
#include <highgui.h>

/*
 * some common function used for test
 */

/*
 * generate sample points, trainData and response are paired
 */
void genClusterSamples(CvRNG& rng, CvMat*& points, CvMat*& clusters,
		int sample_count, int cluster_count, CvSize img);

/*
 * generate test points
 */
void genTestClusterSamples(CvRNG& rng, CvMat*& points, CvMat*& clusters, int sample_count,
		CvSize img);

/*
 * add trainData to img, with responded color
 */
void postPointImage(CvMat* points, CvMat* clusters, int sample_count,
		IplImage* img, CvScalar *color_tab, int dim=2);

/*
 * release mat memory space, trainData and response are paired
 */
void releaseClusterSamples(CvMat*& points, CvMat*& clusters);

#endif /* COMMONTEST_H_ */
