#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "HashBuckets.h"
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

void foo(Mat img, int r, int c, bool mirror, int rot, int patchLen=5) {
    if (mirror) {
        flip(img, img, 1);
        c = img.cols - c - 1;
    }
    if (rot != -1)
        rotate(img, img, rot);


    Mat imgGx, imgGy;
    spatialGradient(img, imgGx, imgGy);
    imgGx.convertTo(imgGx, CV_64F);
    imgGy.convertTo(imgGy, CV_64F);

    if (rot == ROTATE_90_COUNTERCLOCKWISE) {
        int nc = r;
        r = img.cols - c - 1;
        c = nc;
    } else if (rot == ROTATE_90_CLOCKWISE) {
        int nr = c;
        c = img.rows - r - 1;
        r = nr;
    } else if (rot == ROTATE_180) {
        r = img.rows - r - 1;
        c = img.cols - c - 1;
    }

    Mat patchGx = imgGx(Range(r - patchLen/2, r + patchLen/2 + 1),
                        Range(c - patchLen/2, c + patchLen/2 + 1)).clone();
    cout << r << " " << c << endl;
    debugMat(patchGx);
}

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

    int patchLen = 5;

    int r = patchLen/2 + 100, c = patchLen/2 + 100;

    int rot = ROTATE_90_COUNTERCLOCKWISE;
    bool mirror = true;
//
//    Mat srcCopy;
//    src.copyTo(srcCopy);
//    foo(srcCopy, r, c, mirror, rot, patchLen);

    HashBuckets* h = new HashBuckets(src, 2, patchLen);
    h->breakImg(rot, mirror);


//    imshow("patch", patch);
//    imshow("result1", src);
//    imshow("result2", dst);
//    waitKey(0);
}