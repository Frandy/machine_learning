/*
 * knn.cpp
 *
 *  Created on: Nov 1, 2012
 *      Author: chjd
 */

/*
 * knn, one of the simplest classification algorithm.
 * for a new point, it lookup its k nearest neighbors, and set the new point as the majority.
 */

/*
 * OpenCV中也实现了KNN算法，
 * 这里是OpenCV的doxygen文档关于KNN的部分，
 * http://www710.univ-lyon1.fr/~eguillou/documentation/opencv2/class_cv_k_nearest.html
 * 下面一篇对OpenCV里面KNN源码分析的博文，还没仔细看
 * http://www.aiseminar.cn/bbs/forum.php?mod=viewthread&tid=824
 */

/*
 C interface

 bool train 	( 	const CvMat *  	trainData,
		const CvMat *  	responses,
		const CvMat *  	sampleIdx = 0,
		bool  	is_regression = false,
		int  	maxK = 32,
		bool  	updateBase = false
	)

 float predict( const CvMat* sample
 	 	 [,<params>]
 	)

==>

 float knearest(const CvMat * trainData,
 	 	 	 const CvMat * responses,
 	 	 	 const CvMat* sample,
 	 	 	 int maxK = 32
 	 	 	 )

// 与 kmeans 类似的时，train的输入和kmeans的输入基本上相同，但是kmeans是要得到分类结果，将结果写在response里面，
 * 而knn的train输入的response是有label的，作为训练用的，knn需要训练吗？
 * 与kmeans不同的是，knn应该可以predict，OpenCV里面不是这样给接口的，但这里我就用与其它分类器类似的接口，使用predict
 * 对给定的sample确定分类
 *
 * 这里简化一下，将train与predict合并，

 */

/*
 knn algorithm
 1. calculate the first k-points, make a max-heap by distance,
 2. for each point in the remain train data,
		calculate the distance between samples,
		if it is smaller than the heap-top,
		 	 pop-heap the max distance out,
		 	 push_heap the new neighbor
 3. find the majority label of the k neighbors in the heap

 */

#include "knn.h"

#include "kmeans.h"
// to use the getDis function

bool CmpNeighbor(const Neighbor& a, const Neighbor& b)
{
	return (a.dis<b.dis);
}

void InitKNeighbors(vector<Neighbor>& kneighbors, CvMat* trainData, CvPoint& sample, int maxK)
{
	for(auto i=0;i<maxK;i++)
	{
		CvPoint tmp = cvPointFrom32f(((CvPoint2D32f*)trainData->data.fl)[i]);
		double dis = getDis(tmp,sample);
		Neighbor nneighbor;
		nneighbor.index = i;
		nneighbor.dis = dis;
		kneighbors.push_back(nneighbor);
	}
	make_heap(kneighbors.begin(),kneighbors.end(),CmpNeighbor);
}

void AddNewNeighbor(vector<Neighbor>& kneighbors, CvMat * trainData, int index, CvPoint& sample)
{
	CvPoint tmp = cvPointFrom32f(((CvPoint2D32f*)trainData->data.fl)[index]);
	double dis = getDis(tmp,sample);
	double maxd = kneighbors[0].dis;	// the first one, as the heap-top
	if(dis<maxd)
	{
		pop_heap(kneighbors.begin(),kneighbors.end(),CmpNeighbor);
		kneighbors.pop_back();
		Neighbor nneighbor;
		nneighbor.index = index;
		nneighbor.dis = dis;
		kneighbors.push_back(nneighbor);
		push_heap(kneighbors.begin(),kneighbors.end(),CmpNeighbor);
	}
}

int FindMaxIndex(vector<int>& count, int cluster_count)
{
	int maxk = 0;
	int maxc = count[maxk];
	for(auto i = 0;i<cluster_count;i++)
	{
		if(count[i]>maxc)
			maxk = i;
	}
	return maxk;
}

/*
 * to get the majority label,
 * use majority algorithm ? if no label count larger than maxK/2 ?
 * add the param cluster_count,
 */

int MajorityLabel(vector<Neighbor>& kneighbors,CvMat * responses, int cluster_count)
{
	vector<int> labels_count(cluster_count,0);
	int maxK = kneighbors.size();
	for(auto i=0;i<maxK;i++)
	{
		int label = responses->data.i[kneighbors[i].index];
		labels_count[label] += 1;
	}
	int label_index = FindMaxIndex(labels_count,cluster_count);
	return label_index;
}

int knearest(CvMat * trainData, CvMat * responses, int cluster_count, CvMat* samples, CvMat* sample_responses, int maxK)
{
	int sample_count = samples->rows;
//	assert(sample_count==1);
	int data_count = trainData->rows;
	for(auto i=0;i<sample_count;i++)
	{
		CvPoint sample = cvPointFrom32f(((CvPoint2D32f*)samples->data.fl)[i]);
		vector<Neighbor> kneighbors;
		InitKNeighbors(kneighbors,trainData,sample,maxK);
		for(auto j=maxK;j<data_count;j++)
		{
			AddNewNeighbor(kneighbors,trainData,j,sample);
		}
		int label = MajorityLabel(kneighbors,responses,cluster_count);
		sample_responses->data.i[i] = label;
	}
	return 1;
}


