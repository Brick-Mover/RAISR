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

    string dirPath = "./train_images";
    string outPath = "./result_images";
    string filterPath = "./filters";
    vector<Mat> imageList;
    vector<string> imageNameList;
    readListOfImage(dirPath, imageList, imageNameList);
    RAISR model(imageList, 4, 11, 9);

//    string filterFilePath = "./filters/2018_5_31_19_40_15_scale_2.filter";
//    model.readInFilter(filterFilePath);
    model.train();
    model.writeOutFilter(filterPath);
    dirPath = "./test_images";


    imageList.clear();
    imageNameList.clear();
    readListOfImage(dirPath, imageList, imageNameList);
    vector<Mat> downScaledImageList;
    vector<Mat> cheapScaledImageList;
    vector<Mat> RAISRImageList;
    model.test(true, imageList, downScaledImageList, RAISRImageList, cheapScaledImageList,"None");
    for(int i = 0; i < imageList.size(); i++){
        string currentOutPath = outPath + "/cheapScale_"+ imageNameList[i];
        imwrite(currentOutPath, cheapScaledImageList[i]);

        currentOutPath = outPath + "/RAISR_" + imageNameList[i];
        imwrite(currentOutPath, RAISRImageList[i]);

        currentOutPath = outPath + "/downScaled_" + imageNameList[i];
        imwrite(currentOutPath, downScaledImageList[i]);

    }


}