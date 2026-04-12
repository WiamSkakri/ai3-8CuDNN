// SPDX-License-Identifier: Apache-2.0

#include "exec_cudnn.hpp"
#include <ai3.hpp>
#include <algos.hpp>

template <typename dtype>
Tensor conv2d::winograd_nonfused(Tensor input, const Tensor &kernel,
                                 const std::optional<const Tensor> &bias,
                                 const uint padding_h, const uint padding_w,
                                 const uint stride_h, const uint stride_w,
                                 const uint dilation_h, const uint dilation_w,
                                 const PaddingMode padding_mode, uint groups) {
    errs::bail_if(stride_h != 1 || stride_w != 1,
                  "winograd nonfused not implemented for stride not equal to 1 "
                  "see `Supported Algorithms for cudnnConvolutionForward() 2D "
                  "Convolutions. Filter descriptor wDesc: _NCHW` "
                  "at "
                  "https://docs.nvidia.com/deeplearning/cudnn/latest/api/"
                  "cudnn-cnn-library.html");
    errs::bail_if(
        kernel.width() != 3 || kernel.height() != 3,
        "winograd nonfused not implemented for kernel height and kernel width "
        "not equal to 3 "
        "see `Supported Algorithms for cudnnConvolutionForward() 2D "
        "Convolutions. Filter descriptor wDesc: _NCHW` "
        "at "
        "https://docs.nvidia.com/deeplearning/cudnn/latest/api/"
        "cudnn-cnn-library.html");
    return conv_bias_forward_with_algo<dtype>(
        std::move(input), kernel, bias, padding_h, padding_w, stride_h,
        stride_w, dilation_h, dilation_w, padding_mode, groups,
        CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED);
}

template Tensor conv2d::winograd_nonfused<float>(CONV2D_PARAMS);
template Tensor conv2d::winograd_nonfused<double>(CONV2D_PARAMS);
