#include <algorithm>
#include <cfloat>
#include <string.h>
#include "utils/caffe.h"

namespace caffe {
void im2col_cpu(const float* data_im, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_col) {
  int height_col = (height + 2 * pad_h - kernel_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - kernel_w) / stride_w + 1;
  int channels_col = channels * kernel_h * kernel_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % kernel_w;
    int h_offset = (c / kernel_w) % kernel_h;
    int c_im = c / kernel_h / kernel_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_col[(c * height_col + h) * width_col + w] =
            data_im[(c_im * height + h_pad) * width + w_pad];
        else
          data_col[(c * height_col + h) * width_col + w] = 0;
      }
    }
  }
}

void col2im_cpu(const float* data_col, const int channels,
    const int height, const int width, const int patch_h, const int patch_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* data_im) {
  memset(data_im, 0, height * width * channels * sizeof(float));
  int height_col = (height + 2 * pad_h - patch_h) / stride_h + 1;
  int width_col = (width + 2 * pad_w - patch_w) / stride_w + 1;
  int channels_col = channels * patch_h * patch_w;
  for (int c = 0; c < channels_col; ++c) {
    int w_offset = c % patch_w;
    int h_offset = (c / patch_w) % patch_h;
    int c_im = c / patch_h / patch_w;
    for (int h = 0; h < height_col; ++h) {
      for (int w = 0; w < width_col; ++w) {
        int h_pad = h * stride_h - pad_h + h_offset;
        int w_pad = w * stride_w - pad_w + w_offset;
        if (h_pad >= 0 && h_pad < height && w_pad >= 0 && w_pad < width)
          data_im[(c_im * height + h_pad) * width + w_pad] +=
            data_col[(c * height_col + h) * width_col + w];
      }
    }
  }
}

void forward_max_pooling(const float* bottom, const int num, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* top, float* mask) {
  int top_height = (height + pad_h * 2 -kernel_h ) / stride_h + 1;
  int top_width = (width + pad_w * 2 -kernel_w ) / stride_w + 1;
  int top_count = num * top_height * top_width * channels;
  for (int i = 0; i < top_count; i++) {
    mask[i] = -1;
    top[i] = -FLT_MAX;
  }
  const int bottom_offset =  height * width;
  const int top_offset = top_height * top_width;
  // The main loop
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < top_height; ++ph) {
        for (int pw = 0; pw < top_width; ++pw) {
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int hend = std::min(hstart + kernel_h, height);
          int wend = std::min(wstart + kernel_w, width);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
          const int top_index = ph * top_width + pw;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width + w;
              if (bottom[index] > top[top_index]) {
                top[top_index] = bottom[index];
                mask[top_index] = index;
              }
            }
          }
        }
      }
      // compute offset
      bottom += bottom_offset;
      top += top_offset;
      mask += top_offset;
    }
  }
}

void backward_max_pooling(const float* top, const float* mask, const int num,
    const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* bottom) {
  int top_height = (height + pad_h * 2 -kernel_h ) / stride_h + 1;
  int top_width = (width + pad_w * 2 -kernel_w ) / stride_w + 1;
  const int top_offset = top_height * top_width;
  const int bottom_offset = height * width;
  memset(bottom, 0, sizeof(float) * num * channels * bottom_offset);
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < top_height; ++ph) {
        for (int pw = 0; pw < top_width; ++pw) {
          const int top_idx = ph * top_width + pw;
          const int bottom_idx = static_cast<int>(mask[top_idx]);
          bottom[bottom_idx] += top[top_idx];
        }
      }
      top += top_offset;
      mask += top_offset;
      bottom += bottom_offset;
    }
  }
}

void forward_ave_pooling(const float* bottom, const int num, const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* top) {
  int top_height = (height + pad_h * 2 -kernel_h ) / stride_h + 1;
  int top_width = (width + pad_w * 2 -kernel_w ) / stride_w + 1;
  int top_count = num * top_height * top_width * channels;
  for (int i = 0; i < top_count; i++) {
    top[i] = 0;
  }
  const int bottom_offset =  height * width;
  const int top_offset = top_height * top_width;
  // The main loop
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < top_height; ++ph) {
        for (int pw = 0; pw < top_width; ++pw) {
          int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int hend = std::min(hstart + kernel_h, height+pad_h);
          int wend = std::min(wstart + kernel_w, width+pad_w);
		  int pool_size = (hend-hstart) * (wend-wstart);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
		  hend = std::min(hend, height);
		  wend = std::min(wend, width);
          const int top_index = ph * top_width + pw;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width + w;
			  top[top_index] += bottom[index];
            }
          }
		  top[top_index] /= pool_size;
        }
      }
      // compute offset
      bottom += bottom_offset;
      top += top_offset;
    }
  }
}
void backward_ave_pooling(const float* top, const int num,
    const int channels,
    const int height, const int width, const int kernel_h, const int kernel_w,
    const int pad_h, const int pad_w,
    const int stride_h, const int stride_w,
    float* bottom) {
  int top_height = (height + pad_h * 2 -kernel_h ) / stride_h + 1;
  int top_width = (width + pad_w * 2 -kernel_w ) / stride_w + 1;
  const int top_offset = top_height * top_width;
  const int bottom_offset = height * width;
  memset(bottom, 0, sizeof(float) * num * channels * bottom_offset);
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      for (int ph = 0; ph < top_height; ++ph) {
        for (int pw = 0; pw < top_width; ++pw) {
		  int hstart = ph * stride_h - pad_h;
          int wstart = pw * stride_w - pad_w;
          int hend = std::min(hstart + kernel_h, height+pad_h);
          int wend = std::min(wstart + kernel_w, width+pad_w);
		  int pool_size = (hend-hstart) * (wend-wstart);
          hstart = std::max(hstart, 0);
          wstart = std::max(wstart, 0);
		  hend = std::min(hend, height);
		  wend = std::min(wend, width);
          const int top_index = ph * top_width + pw;
          for (int h = hstart; h < hend; ++h) {
            for (int w = wstart; w < wend; ++w) {
              const int index = h * width + w;
			  bottom[index] += top[top_index] / pool_size;
            }
          }
	
        }
      }
      top += top_offset;
      bottom += bottom_offset;
    }
  }
}
}
