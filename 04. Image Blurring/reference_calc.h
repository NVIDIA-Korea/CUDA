
#ifndef _REFERENCE_CALC_H_
#define _REFERENCE_CALC_H_

#include <cmath>
#include <cassert>
#include "utils.h"

void channelConvolution(const unsigned char* const channel,
                        unsigned char* const channelBlurred,
                        const size_t numRows, const size_t numCols,
                        const float *filter, const int filterWidth);
void referenceCalculation(const uchar4* const rgbaImage, uchar4 *const outputImage,
                          size_t numRows, size_t numCols,
                          const float* const filter, const int filterWidth);


#endif /* _REFERENCE_CALC_H_ */