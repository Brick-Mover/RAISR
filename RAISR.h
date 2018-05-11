//
// Created by Zhen Feng on 5/6/18.
//

#ifndef RAISR_RAISR_H
#define RAISR_RAISR_H

#include "HashBuckets.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"

enum Rotation{
    NO_ROTATION =-1,
    ROTATE_90   = 0,
    ROTATE_180  = 1,
    ROTATE_270  = 2,
};

enum Mirror{
    MIRROR    = 0,
    NO_MIRROR =1,
};

class RAISR {
public:
    RAISR(std::vector<cv::Mat>& imageMatList, int scale, int patchLength, int gradientLength);
    void train();
    void test(std::vector<cv::Mat> &imageMatList, std::vector<cv::Mat>& downScaledImageList, std::vector<cv::Mat>& resultHRImageList, std::vector<cv::Mat>& resultLRImageList);
private:
    bool trained;
    int patchLength;
    int gradientLength;
    int scale;
    std::vector<std::vector<cv::Mat>>  filterBuckets;
    std::vector<cv::Mat>& imageMatList;

};

Rotation& operator++( Rotation &c );
Rotation operator++( Rotation &c, int );
void imageOctuplicate(cv::Mat original, std::vector<cv::Mat> result);
cv::Mat downGrade(cv::Mat image, int scale);
void fillBucketsMatrix(std::vector<std::vector<cv::Mat>>& ATA, std::vector<std::vector<cv::Mat>> & ATb, int hashValue, cv::Mat patch, double HRPixel, int pixelType);
int getHashValue(HashBuckets & buckets, int r, int c, Rotation rotateFlag, Mirror mirror);


#endif //RAISR_RAISR_H
