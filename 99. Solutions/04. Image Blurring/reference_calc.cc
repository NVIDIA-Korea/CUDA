
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "reference_calc.h"
#include <algorithm>
#include <iostream>

using namespace cv;

cv::Mat imageInputRGBA;
cv::Mat imageOutputRGBA;
        
float *h_filter__;

size_t numRows() { return imageInputRGBA.rows; }
size_t numCols() { return imageInputRGBA.cols; }


void channelConvolution(const unsigned char* const channel,
                        unsigned char* const channelBlurred,
                        const size_t numRows, const size_t numCols,
                        const float *filter, const int filterWidth)
{
  //Dealing with an even width filter is trickier
  assert(filterWidth % 2 == 1);

  //For every pixel in the image
  for (int r = 0; r < (int)numRows; ++r) {
    for (int c = 0; c < (int)numCols; ++c) {
      float result = 0.f;
      //For every value in the filter around the pixel (c, r)
      for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) {
        for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) {
          //Find the global image position for this filter position
          //clamp to boundary of the image
          int image_r = std::min(std::max(r + filter_r, 0), static_cast<int>(numRows - 1));
          int image_c = std::min(std::max(c + filter_c, 0), static_cast<int>(numCols - 1));

          float image_value = static_cast<float>(channel[image_r * numCols + image_c]);
          float filter_value = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];

          result += image_value * filter_value;
        }
      }

      channelBlurred[r * numCols + c] = result;
    }
  }
}

void referenceCalculation(const uchar4* const rgbaImage, uchar4 *const outputImage,
                          size_t numRows, size_t numCols,
                          const float* const filter, const int filterWidth)
{
    unsigned char *red   = new unsigned char[numRows * numCols];
    unsigned char *blue  = new unsigned char[numRows * numCols];
    unsigned char *green = new unsigned char[numRows * numCols];

    unsigned char *redBlurred   = new unsigned char[numRows * numCols];
    unsigned char *blueBlurred  = new unsigned char[numRows * numCols];
    unsigned char *greenBlurred = new unsigned char[numRows * numCols];

    //First we separate the incoming RGBA image into three separate channels
    //for Red, Green and Blue
    for (size_t i = 0; i < numRows * numCols; ++i) {
        uchar4 rgba = rgbaImage[i];
        red[i]   = rgba.x;
        green[i] = rgba.y;
        blue[i]  = rgba.z;
    }

    //Now we can do the convolution for each of the color channels
    channelConvolution(red, redBlurred, numRows, numCols, filter, filterWidth);
    channelConvolution(green, greenBlurred, numRows, numCols, filter, filterWidth);
    channelConvolution(blue, blueBlurred, numRows, numCols, filter, filterWidth);

    //now recombine into the output image - Alpha is 255 for no transparency
    for (size_t i = 0; i < numRows * numCols; ++i) {
        uchar4 rgba = make_uchar4(redBlurred[i], greenBlurred[i], blueBlurred[i], 255);
        outputImage[i] = rgba;
    }

    delete[] red;
    delete[] green;
    delete[] blue;

    delete[] redBlurred;
    delete[] greenBlurred;
    delete[] blueBlurred;
}

//return types are void since any internal error will be handled by quitting
//no point in returning error codes...
//returns a pointer to an RGBA version of the input image
//and a pointer to the single channel grey-scale output
//on both the host and device
void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
                float **h_filter, int *filterWidth,
                const std::string &filename) {
    cv::Mat image = cv::imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
    if (image.empty()) {
    std::cerr << "Couldn't open file: " << filename << std::endl;
    exit(1);
    }

    cv::cvtColor(image, imageInputRGBA, CV_BGR2RGBA);

    //allocate memory for the output
    imageOutputRGBA.create(image.rows, image.cols, CV_8UC4);

    //This shouldn't ever happen given the way the images are created
    //at least based upon my limited understanding of OpenCV, but better to check
    if (!imageInputRGBA.isContinuous() || !imageOutputRGBA.isContinuous()) {
        std::cerr << "Images aren't continuous!! Exiting." << std::endl;
        exit(1);
    }

    *h_inputImageRGBA  = (uchar4 *)imageInputRGBA.ptr<unsigned char>(0);
    *h_outputImageRGBA = (uchar4 *)imageOutputRGBA.ptr<unsigned char>(0);

    const size_t numPixels = numRows() * numCols();

    //now create the filter that they will use
    const int blurKernelWidth = 9;
    const float blurKernelSigma = 2.;

    *filterWidth = blurKernelWidth;

    //create and fill the filter we will convolve with
    *h_filter = new float[blurKernelWidth * blurKernelWidth];
    h_filter__ = *h_filter;

    float filterSum = 0.f; //for normalization

    for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
        for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
          float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
          (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
          filterSum += filterValue;
        }
    }

    float normalizationFactor = 1.f / filterSum;

    for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) {
        for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) {
          (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
        }
    }
}

void postProcess(const std::string& output_file) {
    const int numPixels = numRows() * numCols();
    
    cv::Mat imageOutputBGR;
    cv::cvtColor(imageOutputRGBA, imageOutputBGR, CV_RGBA2BGR);
    //output the image
    cv::imwrite(output_file.c_str(), imageOutputBGR);

    //cleanup
    delete[] h_filter__;
}

int main(int argc, char **argv) {
    uchar4 *h_inputImageRGBA;
    uchar4 *h_outputImageRGBA;

    float *h_filter;
    int    filterWidth;

    std::string input_file;
    std::string output_file;
    if (argc == 3) {
        input_file  = std::string(argv[1]);
        output_file = std::string(argv[2]);
    }
    else {
        std::cerr << "Usage: ./blur input_file output_file" << std::endl;
        exit(1);
    }
    //load the image and give us our input and output pointers
    preProcess(&h_inputImageRGBA, &h_outputImageRGBA,
             &h_filter, &filterWidth, input_file);

    referenceCalculation(h_inputImageRGBA, h_outputImageRGBA, numRows(), numCols(), h_filter, filterWidth);

    //check results and output the blurred image
    postProcess(output_file);
    
    return 0;
}