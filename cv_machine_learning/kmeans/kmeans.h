/*
 * kmeans.h
 *
 *  Created on: Oct 31, 2012
 *      Author: chjd
 */

#ifndef KMEANS_H_
#define KMEANS_H_

#include <cxcore.h>
#include <vector>
using std::vector;


/*
 * the main procedure of k-means algorithm,
 * the interface is nearly the same with cvKMeans2
 * samples, sample points, sample_count x 1, each row is a point.
 * 		and the CvMat type CV_32FC2, so use
 * 			CvPoint pt = cvPointFrom32f(((CvPoint2D32f*)samples->data.fl)[i])
 * labels, response, sample_count x 1, each row is a int
 * 		and the CvMat type CV_32SC1, so use
 * 			labels->data.i[i]
 */
bool myKMeans(CvMat* samples, int cluster_count, CvMat* labels, CvTermCriteria termcrit);

/*
 * the procedure to get range of points
 */
void getMinMax(CvMat* samples,int& minx,int& maxx,int& miny,int& maxy);

/*
 * the procedure to generate one center, in the min-max range of points
 */
CvPoint genRandCenter(int minx,int maxx,int& miny, int& maxy,CvRNG* rng);

/*
 * the procedure to generate k centers
 */
void genInitCenters(int minx,int maxx,int miny, int maxy, CvRNG* rng, int cluster_count,vector<CvPoint>& centers);

/*
 * return the distance of two points
 */
double getDis(CvPoint& x,CvPoint& y);

/*
 * calculate the distance from the point to each centers,
 * take the closest one as its center and return its index as label
 */
int getLabel(CvPoint& tp,vector<CvPoint>& centers);

/*
 * generate all the labels for all the points in samples
 */
void assignLabels(CvMat* samples, int cluster_count, CvMat* labels,vector<CvPoint>& centers);

/*
 * calculate the accurate center for each cluster,
 * and calculate the difference between the new center and the old one
 */
double AverageCenters(CvMat* samples, int cluster_count, CvMat* labels,vector<CvPoint>& centers);

/*
 * in fact, the procedure assignLabels and averageCenters can merge into one procedure,
 * when a point get its label, then its x and y can be added to the total x/y of new center
 */

#endif /* KMEANS_H_ */
