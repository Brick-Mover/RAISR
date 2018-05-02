#include "HashBuckets.h"

using namespace std;
using namespace cv;

static float PI = 3.14159265f;

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
    this->img = move(src);
    buckets.reserve(nBucket);
    spatialGradient(img, imgGx, imgGy);
    convertScaleAbs(imgGx, imgGx);
    convertScaleAbs(imgGy, imgGy);
    bucketCnt.resize(24 * 3 * 3);
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

    float angle = PI / 2;
    if (b != 0) {
        angle += atan2f(L1 - d, c);
    } else if (c != 0) {
        angle += atan2f(b, L1 - a);
    } else if (b == 0 && c == 0) {
        angle += atan2f(1, 0);
    } else {
        assert(false);
    }
    float coherence = ( sqrtf(L1) - sqrtf(L2) ) / ( sqrtf(L1) + sqrtf(L2) );
    float strength = sqrtf(L1);
    int angleIdx = int(angle / ( PI / 24 ));
    angleIdx = angleIdx > 23 ? 23 : (angleIdx < 0 ? 0 : angleIdx);
    int strengthIdx = strength > 0.0001 ? 2 : (strength > 0.001 ? 1 : 0);
    int coherenceIdx = coherence > 0.5 ? 2 : (coherence > 0.25 ? 1 : 0);

    return angleIdx + coherenceIdx * 72 + strengthIdx * 24;
}


void HashBuckets::add(int r, int c)
{
    int index = hash(r, c);
    bucketCnt[index]++;
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
    for (int i = 0; i < bucketCnt.size(); i++) {
        printf("%d: %d\n", i, bucketCnt[i]);
    }
}

Mat* HashBuckets::getImg()
{
    return &img;
}
