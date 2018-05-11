#ifndef RAISR_UTILS_H
#define RAISR_UTILS_H

#include <string>
#include <map>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"


std::string type2str(int type);


void debugMat(cv::Mat m);


template<typename T>
void debug(std::vector<T> v, bool newline=true)
{
    if (v.empty()) {
        std::cout << "{}";
    }
    else {
        std::cout << "{";
        int LEN = v.size();
        for (int i = 0; i < LEN; i++) {
            std::cout << v[i];
            if (i != LEN - 1)
                std::cout << ", ";
        }
        std::cout << "}";
    }
    if (newline)
        std::cout << std::endl;
}


bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2);
void readListOfImage(std::string& dirPath, std::vector<cv::Mat>& imageMatList);
#endif //RAISR_UTILS_H
