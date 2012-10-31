/*
 * kmeans.cpp
 *
 *  Created on: Oct 31, 2012
 *      Author: chjd
 */

// to implement k-means algorithm


/*
 * 在http://docs.opencv.org/modules/core/doc/clustering.html，
 * 给出了OpenCV的KMeans算法的接口，有C和C++的。
 * 在之前使用的示例程序中，使用的是C的接口，所以这里也仿照C的接口来实现。
 *
 */

/*

 C interface

 int cvKMeans2(const CvArr* samples,	// Floating-point matrix of input samples, one row per sample
 	 	 	 int cluster_count, 		// Number of clusters to split the set by
 	 	 	 CvArr* labels,				// Input/output integer array that stores the cluster indices for every sample
 	 	 	 CvTermCriteria termcrit,	// The algorithm termination criteria, that is, the maximum number of iterations and/or the desired accuracy
 	 	 	 int attempts=1,
 	 	 	 CvRNG* rng=0,
 	 	 	 int flags=0,
 	 	 	 CvArr* _centers=0,
 	 	 	 double* compactness=0
 	 	 )

 */

/*

k-means algorithm
1. random select k centers,
2. attach each points to one of the centers, and get k cluster
3. estimate more accurate centers for the k cluster
4. repeat 2-3
5. stop criteria: if k centers nearly not change, or iteration number reach limit

 */

#include <cmath>
#include <iostream>
using std::clog;
using std::endl;


#include <cxcore.h>

#include "kmeans.h"

void getMinMax(CvMat* samples,int& minx,int& maxx,int& miny,int& maxy)
{
	int sample_count = samples->rows;
	CvPoint pt = cvPointFrom32f(((CvPoint2D32f*)samples->data.fl)[0]);
	minx = pt.x;
	maxx = pt.x;
	miny = pt.y;
	maxy = pt.y;
	for(auto i=1;i<sample_count;i++)
	{
		CvPoint pt = cvPointFrom32f(((CvPoint2D32f*)samples->data.fl)[i]);
		if(pt.x<minx)
			minx = pt.x;
		else if(pt.x>maxx)
			maxx = pt.x;

		if(pt.y<miny)
			miny = pt.y;
		else if(pt.y>maxy)
			maxy = pt.y;
	}
}

CvPoint genRandCenter(int minx,int maxx,int& miny, int& maxy,CvRNG* rng)
{
	CvPoint center;
	center.x = cvRandInt(rng)%(maxx-minx)+minx;
	center.y = cvRandInt(rng)%(maxy-miny)+miny;
	return center;
}

void genInitCenters(int minx,int maxx,int miny, int maxy, CvRNG* rng, int cluster_count,vector<CvPoint>& centers)
{
	for(auto i=0;i<cluster_count;i++)
		centers.push_back(genRandCenter(minx,maxx,miny,maxy,rng));
}

double getDis(CvPoint& x,CvPoint& y)
{
	double s = (x.x-y.x)*(x.x-y.x)+(x.y-y.y)*(x.y-y.y);
	return sqrt(s);
}

int getLabel(CvPoint& tp,vector<CvPoint>& centers)
{
	int label = 0;
	double dis = getDis(tp,centers[0]);
	for(auto i=1;i<int(centers.size());i++)
	{
		double td = getDis(tp,centers[i]);
		if(td<dis)
			label = i;
	}
	return label;
}

/*
 * notice that, the labels was saved in labels->data.i,
 * while, points was saved in samples->data.fl
 */

void assignLabels(CvMat* samples, int cluster_count, CvMat* labels,vector<CvPoint>& centers)
{
	int sample_count = samples->rows;
	for (auto i = 0; i < sample_count; i++)
	{
		CvPoint pt = cvPointFrom32f(((CvPoint2D32f*)samples->data.fl)[i]);
		int k = getLabel(pt,centers);
		labels->data.i[i] = k;
	}
}

double AverageCenters(CvMat* samples, int cluster_count, CvMat* labels,vector<CvPoint>& centers)
{
	vector<double> newcenterx(cluster_count,0);
	vector<double> newcentery(cluster_count,0);
	vector<int> single_count(cluster_count,0);
	int sample_count = samples->rows;
	for(auto i=0;i<sample_count;i++)
	{
		int k = labels->data.i[i];
		CvPoint pt = cvPointFrom32f(((CvPoint2D32f*)samples->data.fl)[i]);
		newcenterx[k] +=  pt.x;
		newcentery[k] +=  pt.y;
		single_count[k] += 1;
	}
	double eps = 0;
	for(auto i=0;i<cluster_count;i++)
	{
		CvPoint nc;
		nc.x = 0;
		nc.y = 0;
		if(single_count[i]>0)
		{
			nc.x = newcenterx[i]/single_count[i];
			nc.y = newcentery[i]/single_count[i];
		}
		eps += abs(centers[i].x-nc.x)+abs(centers[i].y-nc.y);
		centers[i] = nc;
	}
	return eps;
}

bool myKMeans(CvMat* samples, int cluster_count, CvMat* labels, CvTermCriteria termcrit)
{
	assert(samples->cols==1 && labels->cols==1 && samples->rows==labels->rows);
	int minx,maxx,miny,maxy;
	CvRNG rng = cvRNG(0x0ff0ff0f);
//	clog << "to get min max" << endl;
	getMinMax(samples,minx,maxx,miny,maxy);
//	clog << "to get init centers" << endl;
	vector<CvPoint> centers;
	genInitCenters(minx,maxx,miny,maxy,&rng,cluster_count,centers);
	int limitnum = termcrit.max_iter;
	double eps = termcrit.epsilon;
	int i=0;
	bool ok = false;
//	clog << "begin iteration" << endl;
	while(!ok && i<limitnum)
	{
//		clog << "\t iter. " << i << endl;
//		clog << "to assign labels" << endl;
		assignLabels(samples,cluster_count,labels,centers);
//		clog << "to re-average centers" << endl;
		double teps = AverageCenters(samples,cluster_count,labels,centers);
//		clog << "teps: " << teps << endl;
		if(teps<eps)
			ok = true;
		i++;
	}
	return ok;
}
