#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "HashBuckets.h"
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

#define NNEIGHBOURS 3   // the length of patch's side, must be an odd number.

inline void debug(Mat m)
{
    cout << "Rows: " << m.rows << ", Cols: " << m.cols << endl;
    imshow("test image", m);
}

// consider the n x n neighbors of each pixel, and cluster them
void breakImg(HashBuckets* h, const int n)
{
    for (int r = n/2; r + n/2 < h->getImg()->rows; r++) {
        for (int c = n/2; c + n/2 < h->getImg()->cols; c++) {
            h->add(r - n/2, r + n/2, c - n/2, c + n/2);
        }
    }
}

int main(int argc, char** argv)
{
    Mat src = imread("./imgs/Lenna.png", 0);
    if (!src.data) { printf("Read input image error！ \n"); return false; }

    Mat dst = src;
    Mat patch = src(Rect(0, 0, 5, 5));
    resize(src, dst, Size(), 2, 2, INTER_LINEAR);

    HashBuckets* h = new HashBuckets(src);
    breakImg(h, NNEIGHBOURS);
//    imshow("patch", patch);
//    imshow("result1", src);
//    imshow("result2", dst);
//    waitKey(0);
}