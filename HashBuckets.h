#ifndef TEST_HASHBUCKETS_H
#define TEST_HASHBUCKETS_H

#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"

// The clustering method based on hashing of image structure
class HashBuckets {
public:
    HashBuckets(cv::Mat img, int nBucket=4);
    void add(int r, int c);   // add the image in range to one bucket
    int hash(int r, int c);   // get the hash value of the image in range
    cv::Mat* getImg();
    void breakImg();        // break the image into patches

    static const int PATCHLEN;  // MUST BE AN ODD NUMBER

private:
    // Methods to calculate local gradient properties
    float angle();
    float strength();
    float coherence();

    // Members(images, image gradients, buckets, etc.)
    cv::Mat img;    // the original image
    cv::Mat imgGx;  // the image's gradient in the horizontal direction
    cv::Mat imgGy;  // the image's gradient in the vertical direction
    int nBucket;    // the number of buckets, should be divisible by 4
    std::vector<std::vector<cv::Mat>> buckets;  // the buckets to store the images
};


#endif //TEST_HASHBUCKETS_H