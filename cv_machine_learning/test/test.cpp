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
		int i, sample_count = cvRandInt(&rng)%1000+10;

		CvMat* points;	// trainData
		CvMat* clusters; // response
		// gen samples
		genClusterSamples(rng,points,clusters,sample_count,cluster_count,cvSize(img->width,img->height));

		printf("cluster_count: %d\nsample_count:%d\n\n",cluster_count,sample_count);
/*
		cvKMeans2(points,cluster_count,clusters,
				cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,100,0.01));
*/
		myKMeans(points,cluster_count,clusters,
						cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,100,1));

//		printf("my k-means done.\n");
		//clog << "my k-means done. " << endl;

// to test knn
		CvMat* test_samples;
		CvMat* test_responses;
		int test_sample_count = cvRandInt(&rng)%10+1;
		int maxK = 5;
		genTestClusterSamples(rng,test_samples,test_responses,test_sample_count,cvSize(img->width,img->height));

		knearest(points,clusters,cluster_count,test_samples,test_responses,maxK);


		cvZero(img);
		postPointImage(points, clusters, sample_count, img, color_tab);

		postPointImage(test_samples,test_responses,test_sample_count,img,color_tab,8);


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

