#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "HashBuckets.h"
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

    Mat dst;
    resize(src, dst, Size(), 2, 2, INTER_LINEAR);


    HashBuckets* h = new HashBuckets(src, 4, 5);
    h->breakImg();


//    imshow("patch", patch);
//    imshow("result1", src);
//    imshow("result2", dst);
//    waitKey(0);
}