/*!
 * Copyright (c) 2016 by Contributors
 * \file multibox_detection.cu
 * \brief MultiBoxDetection op
 * \author Joshua Zhang
*/
#include "./multibox_detection-inl.h"

#define WARPS_PER_BLOCK 16
#define THREADS_PER_WARP 32

#define MULTIBOX_DETECTION_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

namespace mshadow {
namespace cuda {
template<typename DType>
__device__ void Clip(DType *value, DType lower, DType upper) {
  if ((*value) < lower) *value = lower;
  if ((*value) > upper) *value = upper;
}

template<typename DType>
__global__ void MergePredictions(DType *out, const DType *cls_prob,
                                 const DType *loc_pred, const DType *anchors,
                                 int num_classes, int num_anchors,
                                 int num_batches, float threshold, bool clip,
                                 float vx, float vy, float vw, float vh) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_batches * num_anchors) return;
  for (int i = index; i < num_batches * num_anchors; i += blockDim.x * gridDim.x) {
    int n_batch = i / num_anchors;
    int n_anchor = i % num_anchors;
    const DType *p_cls_prob = cls_prob + n_batch * num_classes * num_anchors;
    const DType *p_loc_pred = loc_pred + n_batch * num_anchors * 4;
    DType *p_out = out + n_batch * num_anchors * 6;
    DType score = p_cls_prob[n_anchor];
    int id = 0;
    for (int j = 1; j < num_classes; ++j) {
      DType temp = p_cls_prob[j * num_anchors + n_anchor];
      if (temp > score) {
        score = temp;
        id = j;
      }
    }
    if (id > 0 && score < threshold) {
      id = 0;
    }
    p_out[n_anchor * 6] = id - 1;  // restore original class id
    p_out[n_anchor * 6 + 1] = (id == 0 ? DType(-1) : score);
    int offset = n_anchor * 4;
    DType al = anchors[offset];
    DType at = anchors[offset + 1];
    DType ar = anchors[offset + 2];
    DType ab = anchors[offset + 3];
    DType aw = ar - al;
    DType ah = ab - at;
    DType ax = (al + ar) / 2.f;
    DType ay = (at + ab) / 2.f;
    DType ox = p_loc_pred[offset] * vx * aw + ax;
    DType oy = p_loc_pred[offset + 1] * vy * ah + ay;
    DType ow = exp(p_loc_pred[offset + 2] * vw) * aw / 2;
    DType oh = exp(p_loc_pred[offset + 3] * vh) * ah / 2;
    DType xmin = ox - ow;
    DType ymin = oy - oh;
    DType xmax = ox + ow;
    DType ymax = oy + oh;
    if (clip) {
      Clip(&xmin, DType(0), DType(1));
      Clip(&ymin, DType(0), DType(1));
      Clip(&xmax, DType(0), DType(1));
      Clip(&ymax, DType(0), DType(1));
    }
    p_out[n_anchor * 6 + 2] = xmin;
    p_out[n_anchor * 6 + 3] = ymin;
    p_out[n_anchor * 6 + 4] = xmax;
    p_out[n_anchor * 6 + 5] = ymax;
  }
}

template<typename DType>
__global__ void MergeSortDescend(DType *src, DType *dst, int size,
                                 int width, int slices, int step, int offset) {
  int index = blockDim.x * blockIdx.x + threadIdx.x;
  int start = width * index * slices;
  for (int slice = 0; slice < slices; ++slice) {
    if (start >= size) break;
    int middle = start + (width >> 1);
    if (middle > size) middle = size;
    int end = start + width;
    if (end > size) end = size;
    int i = start;
    int j = middle;
    for (int k = start; k < end; ++k) {
      DType score_i = i < size ? src[i * step + offset] : DType(-1);
      DType score_j = j < size ? src[j * step + offset] : DType(-1);
      if (i < middle && (j >= end || score_i > score_j)) {
        for (int n = 0; n < step; ++n) {
          dst[k * step + n] = src[i * step + n];
        }
        ++i;
      } else {
        for (int n = 0; n < step; ++n) {
          dst[k * step + n] = src[j * step + n];
        }
        ++j;
      }
    }
    start += width;
  }
}

template<typename DType>
__device__ void CalculateOverlap(const DType *a, const DType *b, DType *iou) {
  DType w = max(DType(0), min(a[2], b[2]) - max(a[0], b[0]));
  DType h = max(DType(0), min(a[3], b[3]) - max(a[1], b[1]));
  DType i = w * h;
  DType u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
  (*iou) =  u <= 0.f ? static_cast<DType>(0) : static_cast<DType>(i / u);
}

template<typename DType>
__global__ void ApplyNMS(DType *out, int pos, int num_anchors,
                         int step, int id_index, int loc_index,
                         bool force_suppress, float nms_threshold) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  DType compare_id = out[pos * step + id_index];
  if (compare_id < 0) return;  // not a valid positive detection, skip
  DType *compare_loc_ptr = out + pos * step + loc_index;
  for (int i = index; i < num_anchors; i += blockDim.x * gridDim.x) {
    if (i <= pos) continue;
    DType class_id = out[i * step + id_index];
    if (class_id < 0) continue;
    if (force_suppress || (class_id == compare_id)) {
      DType iou;
      CalculateOverlap(compare_loc_ptr, out + i * step + loc_index, &iou);
      if (iou >= nms_threshold) {
        out[i * step + id_index] = -1;
      }
    }
  }
}
}  // namespace cuda

template<typename DType>
inline void MultiBoxDetectionForward(const Tensor<gpu, 3, DType> &out,
                                     const Tensor<gpu, 3, DType> &cls_prob,
                                     const Tensor<gpu, 2, DType> &loc_pred,
                                     const Tensor<gpu, 2, DType> &anchors,
                                     float threshold, bool clip,
                                     const std::vector<float> &variances) {
  CHECK_EQ(variances.size(), 4) << "Variance size must be 4";
  int num_classes = cls_prob.size(1);
  int num_anchors = cls_prob.size(2);
  int num_batches = cls_prob.size(0);
  const int num_threads = THREADS_PER_WARP * WARPS_PER_BLOCK;
  int num_samples = num_batches * num_anchors;
  int num_blocks = (num_samples - 1) / num_threads + 1;
  cuda::MergePredictions<<<num_blocks, num_threads>>>(out.dptr_, cls_prob.dptr_,
    loc_pred.dptr_, anchors.dptr_, num_classes, num_anchors, num_batches,
    threshold, clip, variances[0], variances[1], variances[2], variances[3]);
  MULTIBOX_DETECTION_CUDA_CHECK(cudaPeekAtLastError());
}

template<typename DType>
inline void NonMaximumSuppression(const Tensor<gpu, 3, DType> &out,
                                  const Tensor<gpu, 3, DType> &temp_space,
                                  float nms_threshold, bool force_suppress) {
  int num_anchors = out.size(1);
  int total_threads = num_anchors / 2 + 1;
  const int num_threads = WARPS_PER_BLOCK * THREADS_PER_WARP;
  int num_blocks = (total_threads - 1) / num_threads + 1;
  // sort detection results
  for (int nbatch = 0; nbatch < out.size(0); ++nbatch) {
    DType *src_ptr = out.dptr_ + nbatch * num_anchors * 6;
    DType *dst_ptr = temp_space.dptr_ + nbatch * num_anchors * 6;
    DType *src = src_ptr;
    DType *dst = dst_ptr;
    for (int width = 2; width < (num_anchors << 1); width <<= 1) {
      int slices = (num_anchors - 1) / (total_threads * width) + 1;
      cuda::MergeSortDescend<<<num_blocks, num_threads>>>(src, dst, num_anchors,
        width, slices, 6, 1);
      MULTIBOX_DETECTION_CUDA_CHECK(cudaPeekAtLastError());
      src = src == src_ptr? dst_ptr : src_ptr;
      dst = dst == src_ptr? dst_ptr : src_ptr;
    }
  }
  // apply nms
  num_blocks = (num_anchors - 1) / num_threads + 1;
  for (int nbatch = 0; nbatch < out.size(0); ++nbatch) {
    DType *ptr = out.dptr_ + nbatch * num_anchors * 6;
    for (int pos = 0; pos < num_anchors; ++pos) {
      // suppress against position: pos
      cuda::ApplyNMS<<<num_blocks, num_threads>>>(ptr, pos, num_anchors,
        6, 0, 2, force_suppress, nms_threshold);
      MULTIBOX_DETECTION_CUDA_CHECK(cudaPeekAtLastError());
    }
  }
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(MultiBoxDetectionParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MultiBoxDetectionOp<gpu, DType>(param);
  });
  return op;
}
}  // namespace op
}  // namespace mxnet
