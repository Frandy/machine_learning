/*
 * commontest.cpp
 *
 *  Created on: Nov 1, 2012
 *      Author: chjd
 */

#include "commontest.h"

#include <cstdio>

void genClusterSamples(CvRNG& rng, CvMat*& points, CvMat*& clusters,
		int sample_count, int cluster_count, CvSize img)
{
	points = cvCreateMat(sample_count, 1, CV_32FC2);
	clusters = cvCreateMat(sample_count, 1, CV_32SC1);

	int k = 0, i = 0;
	/* generate random samples */
	/* gaussian distribution */
	for (k = 0; k < cluster_count; k++)
	{
		CvPoint center;
		CvMat point_chunk;
		center.x = cvRandInt(&rng) % img.width;
		center.y = cvRandInt(&rng) % img.height;
		cvGetRows(
				points,
				&point_chunk,
				k * sample_count / cluster_count,
				k == cluster_count - 1 ? sample_count : (k + 1) * sample_count
						/ cluster_count);
		cvRandArr(&rng, &point_chunk, CV_RAND_NORMAL,
				cvScalar(center.x, center.y, 0, 0),
				cvScalar(img.width / 6, img.height / 6, 0, 0));
	}

	/* shuffle samples */
	for (i = 0; i < sample_count / 2; i++)
	{
		CvPoint2D32f* pt1 = (CvPoint2D32f*) points->data.fl + cvRandInt(&rng)
				% sample_count;
		CvPoint2D32f* pt2 = (CvPoint2D32f*) points->data.fl + cvRandInt(&rng)
				% sample_count;
		CvPoint2D32f temp;
		CV_SWAP(*pt1, *pt2, temp);
	}
}

void genTestClusterSamples(CvRNG& rng, CvMat*& points, CvMat*& clusters,
		int sample_count, CvSize img)
{
	points = cvCreateMat(sample_count, 1, CV_32FC2);
	clusters = cvCreateMat(sample_count, 1, CV_32SC1);
	int k = 0, i = 0;
	/* generate random samples */
	/* gaussian distribution */
	for (k = 0; k < sample_count; k++)
	{
		CvPoint center;
		center.x = cvRandInt(&rng) % img.width;
		center.y = cvRandInt(&rng) % img.height;
		//CV_MAT_ELEM((*points),CvPoint,k,0) = center;
		float* p = points->data.fl + k * (points->step / 4);
		*p = center.x;
		*(p + 1) = center.y;
		//printf("test point: (%d,%d)\n",center.x,center.y);
	}
}

void postPointImage(CvMat* points, CvMat* clusters, int sample_count,
		IplImage* img, CvScalar *color_tab, int dim)
{
	for (int i = 0; i < sample_count; i++)
	{
		CvPoint2D32f pt = ((CvPoint2D32f*) points->data.fl)[i];
		int cluster_idx = clusters->data.i[i];
		if (dim <= 2)
			cvCircle(img, cvPointFrom32f(pt), dim, color_tab[cluster_idx],
					CV_FILLED);// CV_FILLED
		else
			cvCircle(img, cvPointFrom32f(pt), dim, color_tab[cluster_idx], 1);
	}
}

void releaseClusterSamples(CvMat*& points, CvMat*& clusters)
{
	cvReleaseMat(&points);
	cvReleaseMat(&clusters);
}
