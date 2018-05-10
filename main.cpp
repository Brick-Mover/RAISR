#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "HashBuckets.h"
#include "RAISR.h"
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

array<int,2> foo() {
    return {1,2};
};

int main(int argc, char** argv)
{
    Mat src = imread("./imgs/im1.jpg", 0);
    if (!src.data)
    {
        printf("Read input image error! \n");
        return -1;
    }
//    Mat rotateImage;
//    Mat filppedImage;
//    rotate(src, rotateImage, ROTATE_90_CLOCKWISE);
//    flip(rotateImage, filppedImage, 1);
//    imshow("original", src);
//    imshow("rotated", rotateImage );
//    imshow("filppedImage", filppedImage);
//    waitKey(0);

    Mat dst;
    resize(src, dst, Size(), 2, 2, INTER_LINEAR);


    HashBuckets* h = new HashBuckets(src, 4, 3);
    h->breakImg();


//    Mat dst;
//    resize(src, dst, Size(), 2, 2, INTER_LINEAR);
//
//
//    HashBuckets* h = new HashBuckets(src, 4, 5);
//    h->breakImg();

}