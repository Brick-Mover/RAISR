#include <assert.h>
#include "RAISR.h"
using namespace std;
using namespace cv;


RAISR::RAISR(vector<Mat> &imageMatList, int scale, int patchLength, int gradientLength):
        trained(false),
        imageMatList(imageMatList),
        scale(scale),
        patchLength(patchLength),
        gradientLength(gradientLength),
        filterBuckets(HashBuckets::numOfAngle* HashBuckets::numOfCoherence* HashBuckets::numOfStrength){
    // we prefer the patchLength and gradientLength are both odd positive number
    // patchLength has to be greater than gradientLength
    // otherwise we can't calculate the gradient
    assert(patchLength%2 == 1
           && gradientLength %2 ==1
           && patchLength >0
           && gradientLength >0
           &&patchLength>= gradientLength);

    // initialize the filters
    int numberOfFilters = scale*scale;
    for(int i =0 ; i< this->filterBuckets.size(); i++){
        filterBuckets[i].resize(numberOfFilters);
    }
}


void RAISR::train() {
    // initialize the calculation buckets
    int numberOfFilters = scale*scale;
    int margin = patchLength/2;
    vector<vector<Mat>> ATA(filterBuckets.size());
    vector<vector<Mat>> ATb(filterBuckets.size());
    for(int i =0 ; i< filterBuckets.size(); i++){
        ATA[i].resize(numberOfFilters);
        ATb[i].resize(numberOfFilters);
    }

    for(int i=0 ; i< imageMatList.size() ; i++){

        Mat HRImage = imageMatList[i];
        Mat LRImage = downGrade(HRImage, scale);

        HashBuckets buckets(LRImage.clone(), (unsigned) scale,(unsigned) gradientLength);

        LRImage.convertTo(LRImage, CV_64F);
        HRImage.convertTo(HRImage, CV_64F);
        int rows = LRImage.rows;
        int cols = LRImage.cols;
        cout << rows << ":" << cols << endl << flush;
        for (int r = margin; r<= rows - margin -1; r++){
            for (int c = margin ; c <= cols - margin -1 ; c++){
//                cout << "r : " << r <<"," << "c : " << c <<endl << flush;
                int pixelType = ((r-margin) % scale) * scale  + ((c-margin) % scale);
                double HRPixel = HRImage.at<double>(r,c);

                // Range is left inclusive and right exclusive function
                Mat patch = LRImage(
                        Range(r - margin, r + margin + 1),
                        Range(c - margin, c + margin + 1)
                ).clone();

                for (Mirror mirrorFlag: {Mirror::NO_MIRROR, Mirror::MIRROR}){
                    if (mirrorFlag == MIRROR) flip(patch, patch, 1);
                    for (Rotation rotateFlag: {Rotation::NO_ROTATION, Rotation::ROTATE_90, Rotation::ROTATE_180, Rotation::ROTATE_270}) {
                        Mat rotatedPatch;
                        if (rotateFlag == Rotation::NO_ROTATION) rotatedPatch = patch.clone();
                        else rotate(patch, rotatedPatch, rotateFlag);
                        int hashValue = getHashValue(buckets, r, c, rotateFlag, mirrorFlag);
                        fillBucketsMatrix(ATA, ATb, hashValue, rotatedPatch, HRPixel, pixelType);
                    }
                }
            }
        }
    }

    for (int i = 0 ; i< filterBuckets.size(); i++){
        for (int j = 0 ; j< numberOfFilters; j++){
            Mat currentEntryFilter;
            if (ATA[i][j].empty()) continue;
            solve(ATA[i][j], ATb[i][j], currentEntryFilter, DECOMP_SVD);
//            currentEntryFilter = conjugateGradientSolver(ATA[i][j], ATb[i][j]);
//            cout << "i: "<< i << " j: "<< j << endl;
            filterBuckets[i][j] = currentEntryFilter;
        }
    }
    trained = true;
}

void RAISR::test(vector<Mat> &imageMatList, vector<Mat> & downScaledImageList,vector<Mat>& resultHRImageList, vector<Mat> &resultLRImageList) {
    if (not trained){
        cout << "you must train the model before test the model"<<endl;
        exit(EXIT_FAILURE);
    }

    for (int i= 0 ; i < imageMatList.size(); i++){
        Mat image = imageMatList[i];
        int rows = image.rows;
        int cols = image.cols;
        int margin = patchLength/2;

        cout << "rows: " << rows << " cols :" << cols << endl;

        // down-scale image
        Mat downScaledImage;
        Size ImageSize = Size(rows/scale, cols/scale);
        resize(image, downScaledImage, ImageSize, 0, 0, INTER_CUBIC);
        downScaledImageList.push_back(downScaledImage);

        // cheap upscale
        Mat LRImage;
        ImageSize = Size(rows, cols);
        resize(downScaledImage, LRImage, ImageSize, 0, 0, INTER_LINEAR);
        resultLRImageList.push_back(LRImage.clone());
        HashBuckets buckets(LRImage.clone(), (unsigned) scale,(unsigned) gradientLength);

        // now convert the LRImage into modifiable one
        LRImage.convertTo(LRImage, CV_64F);

        Mat HRImage = LRImage.clone();

        for (int r = margin; r<= rows - margin -1; r++) {
            for (int c = margin; c <= cols - margin - 1; c++) {
                int pixelType = ((r-margin) % scale) * scale  + ((c-margin) % scale);
                if (r == 485 && c == 5 ){
                    cout << "p";
                }
                Mat patch = LRImage(
                        Range(r - margin, r + margin + 1),
                        Range(c - margin, c + margin + 1)
                ).clone();

                Mat flattedPatch = patch.reshape(0,1);

                int hashValue = getHashValue(buckets, r, c, NO_ROTATION, NO_MIRROR);
                if (filterBuckets[hashValue][pixelType].empty()){
                    continue;
                }
                Mat filteredPixel = flattedPatch*filterBuckets[hashValue][pixelType];

//                cout << HRImage.at<double>(r,c) << endl;
//                cout << filteredPixel.at<double>(0,0) << endl;
//                cout << "r : " << r << " c : "<< c <<endl;
//                cout << HRImage.rows << "     " << HRImage.cols << endl;

                HRImage.at<double>(r,c) = filteredPixel.at<double>(0,0);

//                cout << HRImage.at<double>(r,c) << endl;
            }
        }

        Mat resultHRImage;
        convertScaleAbs(HRImage, resultHRImage);
        resultHRImageList.push_back(resultHRImage);
    }

}


void imageOctuplicate(Mat original, vector<Mat> result){
    Mat filppedImage;
    result.push_back(original);
    flip(original, filppedImage, 1);
    result.push_back(filppedImage);
    for (int i = 0; i< 3 ; i++){
        Mat rotatedImage, newFilppedImage;
        rotate(result[i*2],rotatedImage, ROTATE_90_CLOCKWISE);
        flip(rotatedImage, newFilppedImage,1);
        result.push_back(rotatedImage);
        result.push_back(newFilppedImage);
    }
}

Mat downGrade(Mat image, int scale){

    Size imageSize = image.size();
    int rows = imageSize.height;
    int cols = imageSize.width;
    Size tempImageSize = Size(rows/scale, cols/scale);
    Mat tempImage, resultImage;

    resize(image, tempImage, tempImageSize, 0, 0, INTER_CUBIC);

    resize(tempImage, resultImage, imageSize, 0,0, INTER_LINEAR);

    return resultImage;

}


void fillBucketsMatrix(vector<vector<Mat>> &ATA, vector<vector<Mat>> & ATb, int hashValue, Mat patch, double HRPixel, int pixelType){

    Mat flattedPatch = patch.reshape(0,1);


    Mat ATAElement = flattedPatch.t()*flattedPatch;

    // fill ATA
    if (ATA[hashValue][pixelType].empty()){
        ATA[hashValue][pixelType] = ATAElement;
    }else{
        ATA[hashValue][pixelType] += ATAElement;
    }

    Mat ATbElement = flattedPatch.t()*HRPixel;
    if (ATb[hashValue][pixelType].empty()){
        ATb[hashValue][pixelType] = ATbElement;
    }else{
        ATb[hashValue][pixelType] += ATbElement;
    }
}

int getHashValue(HashBuckets & buckets, int r, int c, Rotation rotateFlag, Mirror mirror){
    bool mirrorFlag = (mirror == MIRROR);
    array<int, 3> hashVector = buckets.hash(r,c, rotateFlag, mirrorFlag);
    return hashVector[0]*HashBuckets::numOfStrength*HashBuckets::numOfCoherence+
           hashVector[1]*HashBuckets::numOfStrength+
           hashVector[2];
}

// Special behavior for ++Rotation
Rotation& operator++( Rotation &c ) {
    using IntType = typename std::underlying_type<Rotation>::type;
    if(c == Rotation::ROTATE_270)
        c = Rotation ::NO_ROTATION;
    else
        c = static_cast<Rotation >( static_cast<IntType>(c) + 1 );
    return c;
}

// Special behavior for Rotation++
Rotation operator++( Rotation &c, int ) {
    Rotation result = c;
    ++c;
    return result;
}

Mat conjugateGradientSolver(Mat A, Mat b){
    int rows = A.rows;
    int cols = A.cols;
    double sumOfA = sum(A)[0];

    Mat result = Mat(rows ,1, CV_64F, double(0));
    while (sumOfA >= 100){
        if (determinant(A) < 1){
            A = A + Mat::eye(rows, cols, CV_64F)* sumOfA*0.000000005;
        }else{
            result += A.inv() * b;
            break;
        }
    }
    return result;
}

