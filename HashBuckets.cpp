#include "HashBuckets.h"

using namespace std;
using namespace cv;

#define PI           3.14159265358979323846

inline void debug(Mat m)
{
    cout << "Rows: " << m.rows << ", Cols: " << m.cols << endl;
    imshow("test image", m);
    waitKey();
}

const int HashBuckets::PATCHLEN = 5;

HashBuckets::HashBuckets(Mat src, int nBucket)
{
    this->nBucket = nBucket;
    this->img = src;
    buckets.reserve(nBucket);
    spatialGradient(img, imgGx, imgGy);
    convertScaleAbs(imgGx, imgGx);
    convertScaleAbs(imgGy, imgGy);
}

// get the hash value of the range
int HashBuckets::hash(int row, int col)
{
    // number of channels remains the same, reshape to n^2 x 1 matrix
    // need to clone() cause ROI does not have consecutive memory
    Mat patchGx = imgGx(Rect(row, col, PATCHLEN, PATCHLEN)).clone().reshape(0, PATCHLEN * PATCHLEN);
    Mat patchGy = imgGy(Rect(row, col, PATCHLEN, PATCHLEN)).clone().reshape(0, PATCHLEN * PATCHLEN);

    Mat patchGrad;
    hconcat(patchGx, patchGy, patchGrad);   // n x 2 matrix
    patchGrad.convertTo(patchGrad, CV_32F); // convert to float point type for matrix multiplication
    Mat GkTG = patchGrad.t() * patchGrad;  // 2 x 2 gradient matrix of pixel k

    /*  Consider the eigenvalues and eigenvectors of
     *      | a   b |
     *      | c   d |
     * */
    float a = GkTG.at<uchar>(0, 0);
    float b = GkTG.at<uchar>(0, 1);
    float c = GkTG.at<uchar>(1, 0);
    float d = GkTG.at<uchar>(1, 1);
    float T = a + d;
    float D = a * c - b * d;
    float L1 = T/2 + powf(((T * T)/4 - D), 0.5);
    float L2 = T/2 - powf(((T * T)/4 - D), 0.5);

    float angle;
    if (b != 0) {
        angle = atan2f(L1 - d, c);
    } else if (c != 0) {
        angle = atan2f(b, L1 - a);
    } else if (b == 0 && c == 0) {
        angle = atan2f(1, 0);
    } else {
        assert(false);
    }
    angle += PI / 2;
    float coherence = ( sqrtf(L1) - sqrtf(L2) ) / ( sqrtf(L1) + sqrtf(L2) );
    float strength = sqrtf(L1);
    fabs(angle + 0.01) / ( PI / 24 );
    return 0;
}

// the coordinate of the top left vertex of Rect
void HashBuckets::add(int r, int c)
{
    int index = hash(r, c);
//    buckets[index % nBucket].push_back(img(Rect(r, c, PATCHLEN, PATCHLEN)));
}


// consider the n x n neighbors of each pixel, and cluster them
void HashBuckets::breakImg()
{
    for (int r = 0; r + PATCHLEN <= img.rows; r++) {
        for (int c = 0; c + PATCHLEN <= img.cols; c++) {
            this->add(r, c);
        }
    }
}

Mat* HashBuckets::getImg()
{
    return &img;
}
