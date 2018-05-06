#include "HashBuckets.h"

using namespace std;
using namespace cv;

static double PI = 3.141592653;

const int HashBuckets::patchLen = 5;
const double HashBuckets::sigma = 2.0f;


HashBuckets::HashBuckets(Mat src, unsigned scale)
{
    this->scale = scale;
    this->img = move(src);
    this->scale = scale;

    spatialGradient(img, imgGx, imgGy);
    convertScaleAbs(imgGx, imgGx);
    convertScaleAbs(imgGy, imgGy);
    imgGx.convertTo(imgGx, CV_64F);
    imgGy.convertTo(imgGy, CV_64F);

//    Mat t;
//    normalize(imgGy, t, 0, 1, cv::NORM_MINMAX);
//    imshow("test", t);

    Mat k = getGaussianKernel( patchLen, sigma, CV_64F);
    Mat W = k * k.t();      // n x n
    this->W = W.reshape(0, 1);    // convert to 1 x n^2 array

    bucketCnt.resize(24 * 3 * 3);
}


// get the hash value of the range
int HashBuckets::hash(int row, int col, map<int, int>& s)
{
    // number of channels remains the same, reshape to n^2 x 1 matrix
    // need to clone() for ROI does not have consecutive memory
    Mat patchGx = imgGx(Rect(row, col, patchLen, patchLen)).clone().reshape(0, patchLen * patchLen);
    Mat patchGy = imgGy(Rect(row, col, patchLen, patchLen)).clone().reshape(0, patchLen * patchLen);

    Mat patchGrad;
    hconcat(patchGx, patchGy, patchGrad);   // n^2 x 2 matrix

    Mat patchGradT = patchGrad.t();         // equivalent to multiplication by diagonal weight matrix
    patchGradT.row(0) = patchGradT.row(0).mul(W);
    patchGradT.row(1) = patchGradT.row(1).mul(W);
    Mat GkTG = patchGradT * patchGrad;      // 2 x 2 gradient matrix of pixel

    Mat eigenvalues;
    Mat eigenvectors;
    eigen(GkTG, eigenvalues, eigenvectors);
    double L1 = eigenvalues.at<double>(0, 0);
    double L2 = eigenvalues.at<double>(0, 1);
    double angle = atan2(eigenvectors.row(0).at<double>(1), eigenvectors.row(0).at<double>(0));
    if (angle < 0)
        angle += PI;
    double coherence = ( sqrtf(L1) - sqrtf(L2) ) / ( sqrtf(L1) + sqrtf(L2) );
    double strength = sqrtf(L1);
    auto angleIdx = int(angle / ( PI / 24 ));
    angleIdx = angleIdx > 23 ? 23 : (angleIdx < 0 ? 0 : angleIdx);
    int strengthIdx = strength > 12 ? 2 : (strength > 5 ? 1 : 0);
    int coherenceIdx = coherence > 0.75 ? 2 : (coherence > 0.5 ? 1 : 0);

//    debugMat(GkTG);
//    cout << L1 << " " << L2 << endl;
//    cout << angle << " " << strength << " " << coherence << endl;

    return angleIdx + coherenceIdx * 72 + strengthIdx * 24;
}


// consider the n x n neighbors of each pixel, and cluster them
void HashBuckets::breakImg()
{
    map<int, int> strength;

//    this->hash(30, 15, strength);

    for (int r = 0; r + patchLen <= img.rows/4; r++) {
        for (int c = 0; c + patchLen <= img.cols/4; c++) {
            int i = this->hash(r, c, strength);
            bucketCnt[i]++;
        }
    }
    for (int i = 0; i < bucketCnt.size(); i++) {
        printf("%d: %d\n", i, bucketCnt[i]);
    }
//    for (auto p : strength)
//        cout << p.first << " " << p.second << endl;
}
