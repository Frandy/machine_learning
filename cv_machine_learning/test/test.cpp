/*
 * test.cpp
 *
 *  Created on: Oct 31, 2012
 *      Author: chjd
 */

// example in book leanring OpenCV, machine learning, k-means algorithn

// good site for OpenCV examples, http://nashruddin.com/OpenCV_Examples_Part_2

#include <cxcore.h>
#include <highgui.h>

#include "commontest.h"
#include "kmeans.h"
#include "knn.h"

#include <cstdio>


int main(int argc, char** argv)
{
	#define MAX_CLUSTER 5
	CvScalar color_tab[MAX_CLUSTER];
	IplImage* img = cvCreateImage(cvSize(500,500),8,3);
	CvRNG rng = cvRNG(0xffffffff);

	color_tab[0] = CV_RGB(255,0,0);
	color_tab[1] = CV_RGB(0,255,0);
	color_tab[2] = CV_RGB(100,100,255);
	color_tab[3] = CV_RGB(255,0,255);
	color_tab[4] = CV_RGB(255,255,0);

	for(;;)
	{
		int k, cluster_count = cvRandInt(&rng)%MAX_CLUSTER+1;
		int i, sample_count = cvRandInt(&rng)%1000+1;

		CvMat* points;	// trainData
		CvMat* clusters; // response
		// gen samples
		genClusterSamples(points,clusters,sample_count,cluster_count,CvSize(img->width,img->height));

		printf("cluster_count: %d\nsample_count:%d\n\n",cluster_count,sample_count);
/*
		cvKMeans2(points,cluster_count,clusters,
				cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,100,0.01));
*/
		myKMeans(points,cluster_count,clusters,
						cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,100,1));

//		printf("my k-means done.\n");
		//clog << "my k-means done. " << endl;

		cvZero(img);
		for(i=0;i<sample_count;i++)
		{
			CvPoint2D32f pt = ((CvPoint2D32f*)points->data.fl)[i];
			int cluster_idx = clusters->data.i[i];
			cvCircle(img,cvPointFrom32f(pt),2,
					color_tab[cluster_idx],CV_FILLED);
		}

		releaseClusterSamples(points,clusters);

		cvShowImage("clusters",img);

		int key = cvWaitKey(0);
		key = key&0x000000FF;
		//printf("the key pressed:%d\n",key);
		if(key==27)
			break;
	}

	return 0;
}

