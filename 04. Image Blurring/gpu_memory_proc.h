
#ifndef _GPU_MEMORY_PROC_H_
#define _GPU_MEMORY_PROC_H_

cv::Mat imageInputRGBA;
cv::Mat imageOutputRGBA;

uchar4 *d_inputImageRGBA__;
uchar4 *d_outputImageRGBA__;

float *h_filter__;

void preProcess(uchar4 **h_inputImageRGBA, uchar4 **h_outputImageRGBA,
                uchar4 **d_inputImageRGBA, uchar4 **d_outputImageRGBA,
                unsigned char **d_redBlurred,
                unsigned char **d_greenBlurred,
                unsigned char **d_blueBlurred,
                const std::string& filename);

void postProcess(const std::string& output_file);

#endif /* _GPU_MEMORY_PROC_H_ */