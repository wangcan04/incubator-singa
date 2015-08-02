#ifndef SINGA_INCLUDE_UTILS_CAFFE_H_
#define SINGA_INCLUDE_UTILS_CAFFE_H_
namespace caffe {
void im2col_cpu(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_col);
void col2im_cpu(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_im);
void forward_max_pooling(const float* bottom, const int num, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* top, float* mask);
void backward_max_pooling(const float* top, const float* mask, const int num,
    const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* bottom);
void forward_ave_pooling(const float* bottom, const int num, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* top);
void backward_ave_pooling(const float* top, const int num,
    const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* bottom);

}
#endif  //  SINGA_INCLUDE_UTILS_CAFFE_H_
