/*
 * commontest.h
 *
 *  Created on: Nov 1, 2012
 *      Author: chjd
 */

#ifndef COMMONTEST_H_
#define COMMONTEST_H_

#include <cxcore.h>

/*
 * some common function used for test
 */

/*
 * generate sample points, trainData and response are paired
 */
void genClusterSamples(CvMat*& points, CvMat*& clusters, int sample_count,
		int cluster_count, CvSize img);

/*
 * release mat memory space, trainData and response are paired
 */
void releaseClusterSamples(CvMat*& points, CvMat*& clusters);

#endif /* COMMONTEST_H_ */
