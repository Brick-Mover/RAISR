#include <assert.h>
#include "RAISR.h"
using namespace std;
using namespace cv;

/************************************************************
 *  This is the constructor for class RAISR
 *
 *  params: imageList      : image in Mat format that need processing
 *          scale          : the scale factor like 2x or 4x
 *          patchLength    : size of the patch for filtering training at each pixel
 *          gradientLength : size of the patch that used to find gradient
 *  return: void
 */
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


/************************************************************
 *  This is the train function which contains main procedures to
 *  train the RAISR model
 *
 *  params: void
 *  return: void
 */
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
    cout << "training process start "<< endl << flush;

    cout << " "<< imageMatList.size() << " images will be used in training process" <<endl << flush;
    // loop each image
    for(int i=0 ; i< imageMatList.size() ; i++){

        cout << " -- train model with No. " << i+1 << " image "<< endl<< flush;

        // get the Low Resolution and High Resolution image pair
        Mat HRImage = imageMatList[i];
        Mat LRImage = downGrade(HRImage, scale);

        // initialize the HashBuckets
        HashBuckets buckets(LRImage.clone(), (unsigned) scale,(unsigned) gradientLength);

        // convert the image to modifiable one
        LRImage.convertTo(LRImage, CV_64F);
        HRImage.convertTo(HRImage, CV_64F);

        // get rows and columns
        int rows = LRImage.rows;
        int cols = LRImage.cols;

        // loop each High Resolution pixel
        for (int r = margin; r<= rows - margin -1; r++){
            for (int c = margin ; c <= cols - margin -1 ; c++){

                // find the type of the pixel
                int pixelType = ((r-margin) % scale) * scale  + ((c-margin) % scale);

                // get the value of current High Resolution pixel
                double HRPixel = HRImage.at<double>(r,c);

                // find the corresponding Low Resolution patch
                // Range is left inclusive and right exclusive function
                Mat patch = LRImage(
                        Range(r - margin, r + margin + 1),
                        Range(c - margin, c + margin + 1)
                ).clone();

                // for each patch, we can mirror it and rotate it 90/180/270 degree at the same time
                // to get 7 more training patches, which gives us total 8 patches at each pixel
                for (Mirror mirrorFlag: {Mirror::NO_MIRROR, Mirror::MIRROR}){
                    if (mirrorFlag == MIRROR) flip(patch, patch, 1);
                    for (Rotation rotateFlag: {Rotation::NO_ROTATION, Rotation::ROTATE_90, Rotation::ROTATE_180, Rotation::ROTATE_270}) {
                        Mat rotatedPatch;
                        if (rotateFlag == Rotation::NO_ROTATION) rotatedPatch = patch.clone();
                        else rotate(patch, rotatedPatch, rotateFlag);
                        int hashValue = getHashValue(buckets, r, c, rotateFlag, mirrorFlag);

                        // use the current patch to fill the calculation matrix
                        fillBucketsMatrix(ATA, ATb, hashValue, rotatedPatch, HRPixel, pixelType);
                    }
                }
            }
        }
    }

    // loop each buckets and loop every calculation matrix inside
    //  to solve least square to get corresponding filter
    for (int i = 0 ; i< filterBuckets.size(); i++){
        for (int j = 0 ; j< numberOfFilters; j++){
            Mat currentEntryFilter;
            if (ATA[i][j].empty()) continue;
            solve(ATA[i][j], ATb[i][j], currentEntryFilter, DECOMP_SVD);
            filterBuckets[i][j] = currentEntryFilter;
        }
    }

    // when training stage is done, we should set flag to true
    trained = true;

    cout << "training process done "<< endl << flush;
}


/************************************************************
 *  This is the test function which is used to test how the model
 *  works for the test image
 *  notes: since RASIR is used to enhance the image during image upscaling
 *         so we here first downscale the image as true test sample
 *         then we apply our RAISR filters on the test sample to generate
 *         corresponding High Resolution Image.
 *         We do cheap upscale on that true test sample here as well, which is
 *         mainly for comparison purposes
 *  params: imageMatList          : test images in Mat format
 *          downScaledImageList   : downscaled image that is used as true test sample
 *          RAISRImageList        : High Resolution Image by applying learned filters on true test sample
 *          cheapScaledImageList  : Cheap upscaled Image by applying bilinear interpolation
 *  return: void
 */
void RAISR::test(vector<Mat> &imageMatList, vector<Mat> & downScaledImageList, vector<Mat>& RAISRImageList, vector<Mat> &cheapScaledImageList) {

    if (not trained){
        cout << "you must train the model before test the model"<<endl;
        exit(EXIT_FAILURE);
    }

    cout << "test process start "<< endl << flush;

    for (int i= 0 ; i < imageMatList.size(); i++){
        Mat image = imageMatList[i];
        int rows = image.rows;
        int cols = image.cols;
        int margin = patchLength/2;

        // downscale image to generate the true test sample
        Mat downScaledImage;
        Size ImageSize = Size(cols/scale, rows/scale);
        resize(image, downScaledImage, ImageSize, 0, 0, INTER_CUBIC);
        downScaledImageList.push_back(downScaledImage);

        // cheap upscale the image
        Mat LRImage;
        ImageSize = Size(cols, rows);
        resize(downScaledImage, LRImage, ImageSize, 0, 0, INTER_LINEAR);
        cheapScaledImageList.push_back(LRImage.clone());

        // construct the HashBuckets
        HashBuckets buckets(LRImage.clone(), (unsigned) scale,(unsigned) gradientLength);

        // now convert the LRImage into modifiable one
        LRImage.convertTo(LRImage, CV_64F);

        // construct the container to hold RAISR image by copy that cheap upscaled image
        Mat HRImage = LRImage.clone();

        // loop each pixel
        for (int r = margin; r<= rows - margin -1; r++) {
            for (int c = margin; c <= cols - margin - 1; c++) {
                int pixelType = ((r-margin) % scale) * scale  + ((c-margin) % scale);

                // get each pixel's corresponding patch
                Mat patch = LRImage(
                        Range(r - margin, r + margin + 1),
                        Range(c - margin, c + margin + 1)
                ).clone();

                // flatten the patch
                Mat flattedPatch = patch.reshape(0,1);

                // find the hashValue for that patch to find
                // the corresponding filter in the filterBuckets
                int hashValue = getHashValue(buckets, r, c, NO_ROTATION, NO_MIRROR);
                if (filterBuckets[hashValue][pixelType].empty()){
                    continue;
                }

                // map the patch to a new pixel value by apply trained filters
                Mat filteredPixel = flattedPatch*filterBuckets[hashValue][pixelType];

                // assign that calculated pixel value back to the image
                HRImage.at<double>(r,c) = filteredPixel.at<double>(0,0);

            }
        }


        Mat resultHRImage;
        convertScaleAbs(HRImage, resultHRImage);
        RAISRImageList.push_back(resultHRImage);
    }

    cout << "test process done "<< endl << flush;

}

/************************************************************
 *  This function is used to downgrade a image to generate a
 *  same size but blured image
 *  params: image  : original image
 *          scaled : define the extent to which the image should
 *                   be blured
 *  return: Blurred Image
 */
Mat downGrade(Mat image, int scale){

    Size imageSize = image.size();
    int rows = imageSize.height;
    int cols = imageSize.width;
    Size tempImageSize = Size(cols/scale, rows/scale);
    Mat tempImage, resultImage;

    resize(image, tempImage, tempImageSize, 0, 0, INTER_CUBIC);

    resize(tempImage, resultImage, imageSize, 0,0, INTER_LINEAR);

    return resultImage;

}

/************************************************************
 *  This function is used to fill each calculation matrix with
 *  a patch
 *  note: it is hard to explain what ATA and ATb, please refer to
 *        RAISR paper for details
 *  params: ATA       : calculation matrix
 *          ATb       : calculation matrix
 *          hashValue : patch's hashValue
 *          patch     : image patch
 *          HRPixel   : the true pixel that for that patch
 *          PixelType : type of that patch's correspond pixel
 *  return: void
 */
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

/************************************************************
 * This is a wrapper function that map HashBucket's hash
 * function to a single hash value
 *
 *  params: buckets    : HashBuckets object
 *          r, c       : pixel's coordinates
 *          rotateFlag : define the rotation condition
 *          Mirror      : define the mirror condition
 *  return: hashValue
 */
int getHashValue(HashBuckets & buckets, int r, int c, Rotation rotateFlag, Mirror mirror){
    bool mirrorFlag = (mirror == MIRROR);
    array<int, 3> hashVector = buckets.hash(r,c, rotateFlag, mirrorFlag);
    return hashVector[0]*HashBuckets::numOfStrength*HashBuckets::numOfCoherence+
           hashVector[1]*HashBuckets::numOfStrength+
           hashVector[2];
}


/************************************************************
 *  This function overload the ++Rotation behaviour for Rotation
 *
 *  params: Rotation reference
 *  return: Rotation
 */
Rotation& operator++( Rotation &c ) {
    using IntType = typename std::underlying_type<Rotation>::type;
    if(c == Rotation::ROTATE_270)
        c = Rotation ::NO_ROTATION;
    else
        c = static_cast<Rotation >( static_cast<IntType>(c) + 1 );
    return c;
}


/************************************************************
 *  This function overload the Rotation++ behaviour for Rotation
 *
 *  params: Rotation reference
 *  return: Rotation
 */
Rotation operator++( Rotation &c, int ) {
    Rotation result = c;
    ++c;
    return result;
}


/************************************************************
 *  This function is a least square solver implementaion with
 *  using conjugate Gradient algorithm
 *  Note: It is trying to find an X that minimize |AX-b| most
 *
 *  params: A, b : matrices used in the calculation
 *  return: x
 */
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

