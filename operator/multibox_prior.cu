/*!
 * Copyright (c) 2016 by Contributors
 * \file multibox_prior.cu
 * \brief generate multibox prior boxes cuda kernels
 * \author Joshua Zhang
*/

#include "./multibox_prior-inl.h"

#define WARPS_PER_BLOCK 1
#define THREADS_PER_WARP 32

#define MULTIBOXPRIOR_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

namespace mshadow {
namespace cuda {
template<typename DType>
__global__ void AssignPriors(DType *out, float size, float sqrt_ratio, int in_width,
                             int in_height, float step_x, float step_y, int stride, int offset) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= in_width * in_height) return;
  int r = index / in_width;
  int c = index % in_width;
  float center_x = (c + 0.5) * step_x;
  float center_y = (r + 0.5) * step_y;
  float w = size * sqrt_ratio / 2;  // half width
  float h = size / sqrt_ratio / 2;  // half height
  DType *ptr = out + index * stride + 4 * offset;
  *(ptr++) = center_x - w;  // xmin
  *(ptr++) = center_y - h;  // ymin
  *(ptr++) = center_x + w;  // xmax
  *(ptr++) = center_y + h;  // ymax
}
}  // namespace cuda

template<typename DType>
inline void MultiBoxPriorForward(const Tensor<gpu, 2, DType> &out,
                            const std::vector<float> &sizes,
                            const std::vector<float> &ratios,
                            int in_width, int in_height) {
  CHECK_EQ(out.CheckContiguous(), true);
  cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
  DType *out_ptr = out.dptr_;
  float step_x = 1.f / in_width;
  float step_y = 1.f / in_height;
  int num_sizes = static_cast<int>(sizes.size());
  int num_ratios = static_cast<int>(ratios.size());

  int num_thread = THREADS_PER_WARP * WARPS_PER_BLOCK;
  dim3 thread_dim(num_thread);
  dim3 block_dim((in_width * in_height - 1) / num_thread + 1);

  int stride = 4 * (num_sizes + num_ratios - 1);
  int offset = 0;
  // ratio = 1, various sizes
  for (int i = 0; i < num_sizes; ++i) {
    cuda::AssignPriors<DType><<<block_dim, thread_dim, 0, stream>>>(out_ptr,
      sizes[i], 1.f, in_width, in_height, step_x, step_y, stride, offset);
    ++offset;
  }
  MULTIBOXPRIOR_CUDA_CHECK(cudaPeekAtLastError());

  // size = sizes[0], various ratios
  for (int j = 1; j < num_ratios; ++j) {
    cuda::AssignPriors<DType><<<block_dim, thread_dim, 0, stream>>>(out_ptr,
      sizes[0], sqrtf(ratios[j]), in_width, in_height, step_x, step_y, stride, offset);
    ++offset;
  }
  MULTIBOXPRIOR_CUDA_CHECK(cudaPeekAtLastError());
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(MultiBoxPriorParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MultiBoxPriorOp<gpu, DType>(param);
  });
  return op;
}

}  // namespace op
}  // namespace mxnet
