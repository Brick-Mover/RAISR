#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "HashBuckets.h"
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

inline void debug(Mat m)
{
    cout << "Rows: " << m.rows << ", Cols: " << m.cols << endl;
    imshow("test image", m);
    waitKey();
}

int main(int argc, char** argv)
{
    Mat src = imread("./imgs/Lenna.png", 0);
    if (!src.data)
    {
        printf("Read input image error! \n");
        return -1;
    }

    Mat dst;
    resize(src, dst, Size(), 2, 2, INTER_LINEAR);

    HashBuckets* h = new HashBuckets(src);
    h->breakImg();

//    imshow("patch", patch);
//    imshow("result1", src);
//    imshow("result2", dst);
//    waitKey(0);
}