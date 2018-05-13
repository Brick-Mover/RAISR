#ifndef RAISR_RAISR_H
#define RAISR_RAISR_H

#include "HashBuckets.h"
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"

/************************************************************
 * Constant variable declaration
 * Rotation : a flag indicating the degree that rotation process is going to take
 * Mirror   : a flag indicating whether image is mirrored or not
 */
enum Rotation{
    NO_ROTATION   =-1,
    ROTATE_90     = 0,
    ROTATE_180    = 1,
    ROTATE_270    = 2,
};

enum Mirror{
    MIRROR    = 0,
    NO_MIRROR =1,
};


/************************************************************
 * Class RAISR
 * This class contains the implementation of RAISR which is used
 * to enhance image during upscaling.
 *
 * Note:
 * Basic idea is to find a way to map blurred image pixel to its
 * corresponding High Resolution pixel. Given a HR image and
 * blurred image pair, a group of filters will be trained to map
 * each blurred image pixel with its certain neighbor pixels to a new
 * pixel which has as less difference with true HR image pixel as
 * possible. Please refer to the paper if details is needed.
 */
class RAISR {
public:
    RAISR(std::vector<cv::Mat>& imageMatList, int scale, int patchLength, int gradientLength);
    void train();
    void test(std::vector<cv::Mat> &imageMatList, std::vector<cv::Mat>& downScaledImageList, std::vector<cv::Mat>& RAISRImageList, std::vector<cv::Mat>& cheapScaledImageList);
private:
    bool trained; // flag indicating whether model is trained or not
    int patchLength; // length of a patch (patch is a size patchLength x patchLength pixel segment)
    int gradientLength; // length of pixel segment that is used to determine patch's hashValue
    int scale; // factor that describe the extent to which the image is scaled
    std::vector<std::vector<cv::Mat>>  filterBuckets; // contains trained filter
    std::vector<cv::Mat>& imageMatList; // list of images that are used to train the model

};

/************************************************************
 * Module private method declaration
 *
 */
cv::Mat conjugateGradientSolver(cv::Mat A, cv::Mat b);
cv::Mat downGrade(cv::Mat image, int scale);
void fillBucketsMatrix(std::vector<std::vector<cv::Mat>>& ATA, std::vector<std::vector<cv::Mat>> & ATb, int hashValue, cv::Mat patch, double HRPixel, int pixelType);
int getHashValue(HashBuckets & buckets, int r, int c, Rotation rotateFlag, Mirror mirror);
Rotation& operator++( Rotation &c );
Rotation operator++( Rotation &c, int );

#endif //RAISR_RAISR_H
