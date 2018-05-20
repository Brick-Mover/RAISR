#include "opencv2/opencv.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "RAISR.h"
#include <vector>
#include <iostream>

using namespace cv;
using namespace std;

//void foo(Mat img, int r, int c, bool mirror, int rot, int patchLen=5) {
//    if (mirror) {
//        flip(img, img, 1);
//        c = img.cols - c - 1;
//    }
//    if (rot != -1)
//        rotate(img, img, rot);
//
//
//    Mat imgGx, imgGy;
//    spatialGradient(img, imgGx, imgGy);
//    imgGx.convertTo(imgGx, CV_64F);
//    imgGy.convertTo(imgGy, CV_64F);
//
//    if (rot == ROTATE_90_COUNTERCLOCKWISE) {
//        int nc = r;
//        r = img.cols - c - 1;
//        c = nc;
//    } else if (rot == ROTATE_90_CLOCKWISE) {
//        int nr = c;
//        c = img.rows - r - 1;
//        r = nr;
//    } else if (rot == ROTATE_180) {
//        r = img.rows - r - 1;
//        c = img.cols - c - 1;
//    }
//
//    Mat patchGx = imgGx(Range(r - patchLen/2, r + patchLen/2 + 1),
//                        Range(c - patchLen/2, c + patchLen/2 + 1)).clone();
//    cout << r << " " << c << endl;
//    debugMat(patchGx);
//}


int main(int argc, char** argv) {
//    Mat src = imread("./train_images/cat.png", CV_LOAD_IMAGE_COLOR );
//    if (!src.data)
//    {
//        printf("Read input image error! \n");
//        return -1;
//    }
//    Mat tempImage = src.clone();
//    flip(tempImage, tempImage, 1);
//    rotate(tempImage, tempImage, RotateFlags::ROTATE_90_CLOCKWISE);
//
//    Mat anotherImage = src.clone();
//    rotate(anotherImage, anotherImage,RotateFlags::ROTATE_90_CLOCKWISE);
//    flip(anotherImage, anotherImage, 1);
//
//    imshow("oringinal", src);
//    imshow("1", tempImage);
//    imshow("2", anotherImage);
//    waitKey(0);

//
//    Mat dst;
//    resize(src, dst, Size(), 2, 2, INTER_LINEAR);
//
//    int patchLen = 5;
//
//    int r = patchLen/2 + 100, c = patchLen/2 + 100;
//
//    int rot = ROTATE_90_COUNTERCLOCKWISE;
//    bool mirror = true;
//
//    Mat srcCopy;
//    src.copyTo(srcCopy);
//    foo(srcCopy, r, c, mirror, rot, patchLen);

//    HashBuckets* h = new HashBuckets(src, 2, patchLen);
//    h->breakImg(rot, mirror);


//    imshow("patch", patch);
//    imshow("result1", src);
//    imshow("result2", dst);
//    waitKey(0);


//    Mat A = Mat::eye(4, 4, CV_64F);
//    debugMat(A);
    string dirPath = "./train_images";
    vector<Mat> imageList;
    readListOfImage(dirPath, imageList);
    RAISR model(imageList, 2, 11, 9);
    model.train();

    dirPath = "./test_images";
    imageList.clear();
    readListOfImage(dirPath, imageList);
    vector<Mat> downScaledImage;
    vector<Mat> cheapScaledImageList;
    vector<Mat> RAISRImageList;
    model.test(imageList, downScaledImage, RAISRImageList, cheapScaledImageList,"Randomness");
    for(int i = 0; i < imageList.size(); i++){
        imshow("original", imageList[i]);
        imshow("downScaled", downScaledImage[i]);
        imshow("cheapScale", cheapScaledImageList[i]);
        imshow("RAISR", RAISRImageList[i]);

        Mat LRDiff = cheapScaledImageList[i] - imageList[i];
        Mat HRDiff = RAISRImageList[i] - imageList[i];
        Mat enhenceDiff = cheapScaledImageList[i] - RAISRImageList[i];
        double LRDiffvalue = sum(LRDiff)[0];
        double HRDiffvalue = sum(HRDiff)[0];

        cout << "LRPixelDiffValue: "<< LRDiffvalue << " HRPixelDiffValue : "<< HRDiffvalue << endl;

//        debugMat(LRDiff);
//        debugMat(HRDiff);
//        debugMat(enhenceDiff);
        waitKey(0);
    }

}