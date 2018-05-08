#ifndef TEST_HASHBUCKETS_H
#define TEST_HASHBUCKETS_H

#include "Utils.h"
#include <stdexcept>

class HashBuckets {
public:
    HashBuckets(cv::Mat img, unsigned scale, unsigned patchLen);
    int hash(int r, int c, int rot, bool flip);     // get the hash value of the image in range
    void breakImg();            // break the image into patches

    static const double sigma;  // the sigma value of the Gaussian filter

private:
    // Members(images, image gradients, buckets, etc.)
    cv::Mat img;        // the original image, read as CV_8U1
    cv::Mat imgGx;      // the image's gradient in the horizontal direction, CV_64F for matrix operation
    cv::Mat imgGy;      // the image's gradient in the vertical direction, CV_64F for matrix operation
    cv::Mat W;          // the diagonal weighting matrix generated from 2D Gaussian filter, CV_64F
    unsigned scale;     // the scaling ratio
    unsigned patchLen;  // the size of the patch

    std::vector<int> bucketCnt;     // count the number of patches in each bucket
};


#endif //TEST_HASHBUCKETS_H