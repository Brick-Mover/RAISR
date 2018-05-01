#include "HashBuckets.h"

using namespace std;
using namespace cv;

int HashBuckets::hash(int rStart, int rEnd, int cStart, int cEnd)
{
    return 0;
}

void HashBuckets::add(int rStart, int rEnd, int cStart, int cEnd)
{
    int index = hash(rStart, rEnd, cStart, cEnd);
    buckets[index % nBucket].push_back(img(Range(rStart, rEnd), Range(cStart, cEnd)));
}

HashBuckets::HashBuckets(Mat img, int nBucket): nBucket(nBucket), img(img)
{
    buckets.reserve(nBucket);
    spatialGradient(img, imgDx, imgDy);
}

Mat* HashBuckets::getImg()
{
    return &img;
}

void HashBuckets::getGradientStats()
{
    
}