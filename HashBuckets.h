#ifndef TEST_HASHBUCKETS_H
#define TEST_HASHBUCKETS_H

#include "opencv2/opencv.hpp"

// The clustering method based on hashing of image structure
class HashBuckets {
public:
    HashBuckets(cv::Mat img, int nBucket=4);
    void add(int rStart, int rEnd, int cStart, int cEnd);   // add the image in range to one bucket
    int hash(int rStart, int rEnd, int cStart, int cEnd);   // get the hash value of the image in range
    cv::Mat* getImg();

private:
    // Methods to calculate local gradient properties
    void getGradientStats();
    float angle();
    float strength();
    float coherence();

    // Members(images, image gradients, buckets, etc.)
    cv::Mat img;    // the original image
    cv::Mat imgDx;  // the image's derivative in the horizontal direction
    cv::Mat imgDy;  // the image's derivative in the vertical direction
    int nBucket;    // the number of buckets, should be divisible by 4
    std::vector<std::vector<cv::Mat>> buckets;  // the buckets to store the images
};


#endif //TEST_HASHBUCKETS_H