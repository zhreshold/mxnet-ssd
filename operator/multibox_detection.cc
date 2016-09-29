/*!
 * Copyright (c) 2016 by Contributors
 * \file multibox_detection.cc
 * \brief MultiBoxDetection op
 * \author Joshua Zhang
*/
#include "./multibox_detection-inl.h"
#include <algorithm>

namespace mshadow {
template<typename DType>
struct SortElemDescend {
  DType value;
  int index;

  SortElemDescend(DType v, int i) {
    value = v;
    index = i;
  }

  bool operator<(const SortElemDescend &other) const {
    return value > other.value;
  }
};

template<typename DType>
inline void TransformLocations(DType *out, const DType *anchors,
                               const DType *loc_pred, bool clip,
                               float vx, float vy, float vw, float vh) {
  // transform predictions to detection results
  DType al = anchors[0];
  DType at = anchors[1];
  DType ar = anchors[2];
  DType ab = anchors[3];
  DType aw = ar - al;
  DType ah = ab - at;
  DType ax = (al + ar) / 2.f;
  DType ay = (at + ab) / 2.f;
  DType px = loc_pred[0];
  DType py = loc_pred[1];
  DType pw = loc_pred[2];
  DType ph = loc_pred[3];
  DType ox = px * vx * aw + ax;
  DType oy = py * vy * ah + ay;
  DType ow = exp(pw * vw) * aw / 2;
  DType oh = exp(ph * vh) * ah / 2;
  out[0] = clip ? std::max(DType(0), std::min(DType(1), ox - ow)) : (ox - ow);
  out[1] = clip ? std::max(DType(0), std::min(DType(1), oy - oh)) : (oy - oh);
  out[2] = clip ? std::max(DType(0), std::min(DType(1), ox + ow)) : (ox + ow);
  out[3] = clip ? std::max(DType(0), std::min(DType(1), oy + oh)) : (oy + oh);
}

template<typename DType>
inline void MultiBoxDetectionForward(const Tensor<cpu, 3, DType> &out,
                                     const Tensor<cpu, 3, DType> &cls_prob,
                                     const Tensor<cpu, 2, DType> &loc_pred,
                                     const Tensor<cpu, 2, DType> &anchors,
                                     float threshold, bool clip,
                                     const std::vector<float> &variances) {
  CHECK_EQ(variances.size(), 4) << "Variance size must be 4";
  index_t num_classes = cls_prob.size(1);
  index_t num_anchors = cls_prob.size(2);
  const DType *p_anchor = anchors.dptr_;
  for (index_t nbatch = 0; nbatch < cls_prob.size(0); ++nbatch) {
    const DType *p_cls_prob = cls_prob.dptr_ + nbatch * num_classes * num_anchors;
    const DType *p_loc_pred = loc_pred.dptr_ + nbatch * num_anchors * 4;
    DType *p_out = out.dptr_ + nbatch * num_anchors * 6;
    for (index_t i = 0; i < num_anchors; ++i) {
      // find the predicted class id and probability
      DType score = p_cls_prob[i];
      int id = 0;
      for (int j = 1; j < num_classes; ++j) {
        DType temp = p_cls_prob[j * num_anchors + i];
        if (temp > score) {
          score = temp;
          id = j;
        }
      }
      if (id > 0 && score < threshold) {
        id = 0;
      }
      // [id, prob, xmin, ymin, xmax, ymax]
      p_out[i * 6] = id - 1;  // remove background, restore original id
      p_out[i * 6 + 1] = (id == 0 ? DType(-1) : score);
      index_t offset = i * 4;
      TransformLocations(p_out + i * 6 + 2, p_anchor + offset,
        p_loc_pred + offset, clip, variances[0], variances[1],
        variances[2], variances[3]);
    }  // end iter num_anchors
  }  // end iter batch
}

template<typename DType>
inline DType CalculateOverlap(const DType *a, const DType *b) {
  DType w = std::max(DType(0), std::min(a[2], b[2]) - std::max(a[0], b[0]));
  DType h = std::max(DType(0), std::min(a[3], b[3]) - std::max(a[1], b[1]));
  DType i = w * h;
  DType u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
  return u <= 0.f ? static_cast<DType>(0) : static_cast<DType>(i / u);
}

template<typename DType>
inline void NonMaximumSuppression(const Tensor<cpu, 3, DType> &out,
                                  const Tensor<cpu, 3, DType> &temp_space,
                                  float nms_threshold, bool force_suppress) {
  Copy(temp_space, out, out.stream_);
  index_t num_anchors = out.size(1);
  for (index_t nbatch = 0; nbatch < out.size(0); ++nbatch) {
    DType *pout = out.dptr_ + nbatch * num_anchors * 6;
    // sort confidence in descend order
    std::vector<SortElemDescend<DType>> sorter;
    sorter.reserve(num_anchors);
    for (index_t i = 0; i < num_anchors; ++i) {
      DType id = pout[i * 6];
      if (id >= 0) {
        sorter.push_back(SortElemDescend<DType>(pout[i * 6 + 1], i));
      } else {
        sorter.push_back(SortElemDescend<DType>(DType(0), i));
      }
    }
    std::stable_sort(sorter.begin(), sorter.end());
    // re-order output
    DType *ptemp = temp_space.dptr_ + nbatch * num_anchors * 6;
    for (index_t i = 0; i < sorter.size(); ++i) {
      for (index_t j = 0; j < 6; ++j) {
        pout[i * 6 + j] = ptemp[sorter[i].index * 6 + j];
      }
    }
    // apply nms
    for (index_t i = 0; i < num_anchors; ++i) {
      index_t offset_i = i * 6;
      if (pout[offset_i] < 0) continue;  // skip eliminated
      for (index_t j = i + 1; j < num_anchors; ++j) {
        index_t offset_j = j * 6;
        if (pout[offset_j] < 0) continue;  // skip eliminated
        if (force_suppress || (pout[offset_i] == pout[offset_j])) {
          // when foce_suppress == true or class_id equals
          DType iou = CalculateOverlap(pout + offset_i + 2, pout + offset_j + 2);
          if (iou >= nms_threshold) {
            pout[offset_j] = -1;
          }
        }
      }
    }
  }
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(MultiBoxDetectionParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new MultiBoxDetectionOp<cpu, DType>(param);
  });
  return op;
}

Operator* MultiBoxDetectionProp::CreateOperatorEx(Context ctx,
                                                  std::vector<TShape> *in_shape,
                                                  std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK(InferType(in_type, &out_type, &aux_type));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(MultiBoxDetectionParam);
MXNET_REGISTER_OP_PROPERTY(MultiBoxDetection, MultiBoxDetectionProp)
.describe("Convert multibox detection predictions.")
.add_argument("cls_prob", "Symbol", "Class probabilities.")
.add_argument("loc_pred", "Symbol", "Location regression predictions.")
.add_argument("anchors", "Symbol", "Multibox prior anchor boxes")
.add_arguments(MultiBoxDetectionParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
