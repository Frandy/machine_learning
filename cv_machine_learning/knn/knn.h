/*
 * knn.h
 *
 *  Created on: Nov 1, 2012
 *      Author: chjd
 */

#ifndef KNN_H_
#define KNN_H_

#include <cxcore.h>

#include <vector>
using std::vector;

/*
 * the main procedure of k-NN algorithm
 * params:
 * 	trainData,	CV_ROW_SAMPLE, CV_32FC2, CvPoint
 * 	responses,	the labels responses to trainData,
 * 	cluster_count,	number of labels
 * 	samples,	new points to classify
 * 	sample_responses,	to save the labels responses to samples, calculated by this procedure
 * 	maxK,		K neighbors used
 */
int knearest(CvMat * trainData, CvMat * responses, int cluster_count, CvMat* samples, CvMat* sample_responses, int maxK);


/*
 * struct Neighbor, to save the index in trainData and the distance calculated,
 * the distance may be used as weight
 */
struct Neighbor
{
	int index;
	double dis;
};

typedef struct Neighbor Neighbor;

/*
 * less compare operator of Neighbor
 */
bool CmpNeighbor(const Neighbor& a, const Neighbor& b);

/*
 * take the first k points as Neighbors, save them in a max-heap
 */
void InitKNeighbors(vector<Neighbor>& kneighbors, CvMat * trainData, CvPoint& sample, int maxK);

/*
 * for the point in trainData,
 * if the distance is smaller,
 * 		then take it to replace the farthest neighbor in the k-neighbors
 * 		in fact, the farthest one is at the top of the heap
 */
void AddNewNeighbor(vector<Neighbor>& kneighbors, CvMat * trainData, int index, CvPoint& sample);

/*
 * find the max element, return its index
 */
int FindMaxIndex(vector<int>& count, int cluster_count);

/*
 * after get the nearest k neighbors, statistic the k labels,
 * labels are cluster_count integer,
 * the target label is the index of the max element of statistic array
 */
int MajorityLabel(vector<Neighbor>& kneighbors,CvMat * responses, int cluster_count);






#endif /* KNN_H_ */
