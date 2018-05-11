#include "HashBuckets.h"

using namespace std;
using namespace cv;

static double PI = 3.141592653;

const double HashBuckets::sigma = 2.0f;

HashBuckets::HashBuckets(Mat src, unsigned scale, unsigned patchLen) {
    if (patchLen % 2 == 0)
        throw invalid_argument("patch size must be an odd number!");
    this->scale = scale;
    this->img = move(src);
    this->scale = scale;
    this->patchLen = patchLen;

    spatialGradient(img, imgGx, imgGy);
    imgGx.convertTo(imgGx, CV_64F);
    imgGy.convertTo(imgGy, CV_64F);

    Mat k = getGaussianKernel( patchLen, sigma, CV_64F);
    Mat W = k * k.t();      // n x n
    this->W = W.reshape(0, 1);    // convert to 1 x n^2 array
    memset(bucketCnt, 0, sizeof(bucketCnt));
}


// Get the hash value of the patch centered at (r, c)
// rot: 0, 1, 2 for rotate 90/180/270 degrees, any other number will keep the original patch
// mirror: true for mirrored patch
// This allows us to get 8x training examples
array<int, 3> HashBuckets::hash(int r, int c, int rot, bool mirror) {
    // number of channels remains the same, reshape to n^2 x 1 matrix
    // need to clone() for ROI does not have consecutive memory
    Mat patchGx = imgGx(Range(r - patchLen/2, r + patchLen/2 + 1),
                        Range(c - patchLen/2, c + patchLen/2 + 1)).clone();
    Mat patchGy = imgGy(Range(r - patchLen/2, r + patchLen/2 + 1),
                        Range(c - patchLen/2, c + patchLen/2 + 1)).clone();

    // Note the gradient of the rotated image is not the same as the rotated gradient,
    // so we first use geometric relationship to get the new coordinate of the point (r, c).
    // Also, as long as the center of the patch is determined, flip and transpose will not
    // change the result of GTWG. (i.e. the commented section)
    if (mirror) {
        flip(patchGx, patchGx, 1);
        flip(patchGy, patchGy, 1);
        patchGx *= -1;
    }

    if (rot == ROTATE_90_CLOCKWISE) {
        swap(patchGx, patchGy);
//        transpose(patchGx, patchGx);
//        flip(patchGx, patchGx, 1);
        patchGx *= -1;
//        transpose(patchGy, patchGy);
//        flip(patchGy, patchGy, 1);
    } else if (rot == ROTATE_90_COUNTERCLOCKWISE) {
        swap(patchGx, patchGy);
//        transpose(patchGx, patchGx);
//        flip(patchGx, patchGx, 0);
        patchGy *= -1;
//        transpose(patchGy, patchGy);
//        flip(patchGy, patchGy, 0);
    } else if (rot == ROTATE_180) {
//        flip(patchGx, patchGx, 1);
//        flip(patchGx, patchGx, 0);
//        patchGx *= -1;
//        flip(patchGy, patchGy, 1);
//        flip(patchGy, patchGy, 0);
//        patchGy *= -1;
    }

//debugMat(patchGx);

    patchGx = patchGx.reshape(0, patchLen * patchLen);
    patchGy = patchGy.reshape(0, patchLen * patchLen);

    Mat patchGrad;
    hconcat(patchGx, patchGy, patchGrad);   // n^2 x 2 matrix

    Mat patchGradT = patchGrad.t();         // equivalent to multiplication by diagonal weight matrix
    patchGradT.row(0) = patchGradT.row(0).mul(W);
    patchGradT.row(1) = patchGradT.row(1).mul(W);
    Mat GTWG = patchGradT * patchGrad;      // 2 x 2 gradient matrix of pixel
//    debugMat(patchGradT);
    /*  Consider the eigenvalues and eigenvectors of
     *      | a   b |
     *      | c   d |
     * */
    double m_a = GTWG.at<double>(0, 0);
    double m_b = GTWG.at<double>(0, 1);
    double m_c = GTWG.at<double>(1, 0);
    double m_d = GTWG.at<double>(1, 1);
    double T = m_a + m_d;
    double D = m_a * m_d - m_b * m_c;
    double L1 = T/2 + sqrt( (T * T)/4 - D );
    double L2 = T/2 - sqrt( (T * T)/4 - D );

    double angle = 0;
    if (m_b != 0) {
        angle = atan2(L1 - m_d, m_c);
    } else if (c != 0) {
        angle = atan2(m_b, L1 - m_a);
    } else if (m_b == 0 && m_c == 0) {
        angle = atan2(1, 0);
    } else {
        assert(false);
    }
    if (angle < 0)  angle += PI;
    double coherence = ( sqrt(L1) - sqrt(L2) ) / ( sqrt(L1) + sqrt(L2) );
    double strength = sqrt(L1);

    auto angleIdx = int(angle / ( PI / 24 ));

    angleIdx = angleIdx > 23 ? numOfAngle-1 : (angleIdx < 0 ? 0 : angleIdx);
    int strengthIdx = strength > 45 ? numOfStrength-1 : (strength > 30 ? 1 : 0);
    int coherenceIdx = coherence > 0.37 ? numOfStrength-1 : (coherence > 0.21 ? 1 : 0);

    return {angleIdx, coherenceIdx, strengthIdx};
}


// consider the n x n neighbors of each pixel, and cluster them
void HashBuckets::breakImg(int rot, bool mirror) {
    array<int, 3> t;
    for (int r = patchLen/2; r + patchLen/2 < img.rows; r++) {
        for (int c = patchLen/2; c + patchLen/2 < img.cols; c++) {
            for (bool b: { false, true }) {
                t = this->hash(r, c, ROTATE_90_CLOCKWISE, b);
                bucketCnt[t[0]][t[1]][t[2]]++;
                t = this->hash(r, c, ROTATE_90_COUNTERCLOCKWISE, b);
                bucketCnt[t[0]][t[1]][t[2]]++;
                t = this->hash(r, c, ROTATE_180, b);
                bucketCnt[t[0]][t[1]][t[2]]++;
                t = this->hash(r, c, -1, b);
                bucketCnt[t[0]][t[1]][t[2]]++;
            }
        }
    }
    for (int c = 0; c < 3; c++) {
        int cohereCnt = 0;
        for (int s = 0; s < 3; s++) {
            int strCnt = 0;
            for (int a = 0; a < 24; a++) {
                cohereCnt += bucketCnt[a][c][s]; strCnt += bucketCnt[a][c][s];
                printf("%d\t", bucketCnt[a][c][s]);
            }
            printf("\n%d\n", strCnt);
        }
        printf("\n%d\n\n", cohereCnt);
    }
}
