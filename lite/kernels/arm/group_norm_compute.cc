// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "lite/kernels/arm/group_norm_compute.h"
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/op_registry.h"
#include "lite/core/parallel_defines.h"
#include "lite/core/type_system.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {

void GroupNormCompute::PrepareForRun() {}

void GroupNormCompute::Run() {
  auto& param = this->Param<param_t>();
  const float* in = param.x->data<float>();
  const float* scale =
      param.scale == nullptr ? nullptr : param.scale->data<float>();
  const float* bias =
      param.bias == nullptr ? nullptr : param.bias->data<float>();
  float* out = param.out->mutable_data<float>();
  float* saved_mean = param.saved_mean->mutable_data<float>();
  float* saved_variance = param.saved_variance->mutable_data<float>();
  float epsilon = param.epsilon;
  int groups = param.groups;
  int channels = param.channels;
  auto x_dims = param.x->dims();
  int n = x_dims[0];
  int c = x_dims[1];
  if (channels == -1) {
    CHECK_EQ(param.data_layout_str, "NCHW")
        << "it only support NCHW layout!, but recived layout is "
        << param.data_layout_str;
    channels = c;
  }
  int height = x_dims[2];
  int width = x_dims[3];
  int ch_per_group = channels / groups;
  int spatial_size = ch_per_group * height * width;
  int ngroup = n * groups;
  int cnt = spatial_size >> 4;
  int remain = spatial_size % 16;
  float* std_vec = new float[param.saved_variance->numel()];
  // compute saved_mean and saved_variance

  LITE_PARALLEL_BEGIN(n, tid, ngroup) {
    const float* in_p = in + n * spatial_size;
    float sum_spatial = 0.f;
    float summ_spatial = 0.f;
    float32x4_t sum0 = vdupq_n_f32(0.f);
    float32x4_t sum1 = vdupq_n_f32(0.f);
    float32x4_t sum2 = vdupq_n_f32(0.f);
    float32x4_t sum3 = vdupq_n_f32(0.f);
    float32x4_t summ0 = vdupq_n_f32(0.f);
    float32x4_t summ1 = vdupq_n_f32(0.f);
    float32x4_t summ2 = vdupq_n_f32(0.f);
    float32x4_t summ3 = vdupq_n_f32(0.f);
    for (int i = 0; i < cnt; i++) {
      float32x4_t in0 = vld1q_f32(in_p);
      float32x4_t in1 = vld1q_f32(in_p + 4);
      float32x4_t in2 = vld1q_f32(in_p + 8);
      float32x4_t in3 = vld1q_f32(in_p + 12);
      sum0 = vaddq_f32(sum0, in0);
      summ0 = vmlaq_f32(summ0, in0, in0);
      sum1 = vaddq_f32(sum1, in1);
      summ1 = vmlaq_f32(summ1, in1, in1);
      sum2 = vaddq_f32(sum2, in2);
      summ2 = vmlaq_f32(summ2, in2, in2);
      sum3 = vaddq_f32(sum3, in3);
      summ3 = vmlaq_f32(summ3, in3, in3);
      in_p += 16;
    }
    for (int i = 0; i < remain - 3; i += 4) {
      float32x4_t in0 = vld1q_f32(in_p);
      sum1 = vaddq_f32(sum1, in0);
      summ1 = vmlaq_f32(summ1, in0, in0);
      in_p += 4;
    }
    float sum = 0.0;
    float summ = 0.0;
    sum0 = vaddq_f32(sum0, sum1);
    sum2 = vaddq_f32(sum2, sum3);
    summ0 = vaddq_f32(summ0, summ1);
    summ2 = vaddq_f32(summ2, summ3);
    for (int i = 0; i < remain % 4; i++) {
      sum += *in_p;
      summ += (*in_p) * (*in_p);
      in_p++;
    }
    sum0 = vaddq_f32(sum0, sum2);
    summ0 = vaddq_f32(summ0, summ2);
    float32x2_t sum_low = vpadd_f32(vget_low_f32(sum0), vget_high_f32(sum0));
    float32x2_t sum_high = vpadd_f32(vget_low_f32(summ0), vget_high_f32(summ0));
    float32x2_t sum_mix = vpadd_f32(sum_low, sum_high);
    sum += vget_lane_f32(sum_mix, 0);
    summ += vget_lane_f32(sum_mix, 1);
    float mean = sum / spatial_size;
    // float variance = summ / spatial_size - mean * mean;
    // the flolowing code has higher precision than above comment code
    float variance = (summ - mean * mean * spatial_size) / spatial_size;
    float std = 1.f / sqrtf(variance + epsilon);
    saved_mean[n] = mean;
    saved_variance[n] = variance;
    std_vec[n] = std;
  }
  LITE_PARALLEL_END()
  int in_size = height * width;
  cnt = in_size >> 4;
  remain = in_size % 16;
  // compute Group_norm result: out = scale * (in - mean) / std + bias

  LITE_PARALLEL_BEGIN(i, tid, ngroup) {
    const float* in_p = in + i * spatial_size;
    float* out_p = out + i * spatial_size;
    int numc = i % groups;
    numc *= ch_per_group;
    for (int c = 0; c < ch_per_group; c++) {
      int chin = numc + c;
      const float sstd_val =
          (scale == nullptr) ? std_vec[i] : scale[chin] * std_vec[i];
      const float bias_val = (bias == nullptr) ? 0. : bias[chin];
      const float mean_val = saved_mean[i];
      const float32x4_t vsstd = vdupq_n_f32(sstd_val);
      const float32x4_t vbias = vdupq_n_f32(bias_val);
      const float32x4_t vmean = vdupq_n_f32(mean_val);
      for (int k = 0; k < cnt; k++) {
        float32x4_t in0 = vld1q_f32(in_p);
        float32x4_t in1 = vld1q_f32(in_p + 4);
        float32x4_t in2 = vld1q_f32(in_p + 8);
        float32x4_t in3 = vld1q_f32(in_p + 12);
        float32x4_t submean0 = vsubq_f32(in0, vmean);
        float32x4_t submean1 = vsubq_f32(in1, vmean);
        float32x4_t submean2 = vsubq_f32(in2, vmean);
        float32x4_t submean3 = vsubq_f32(in3, vmean);
        float32x4_t out0 = vmlaq_f32(vbias, submean0, vsstd);
        float32x4_t out1 = vmlaq_f32(vbias, submean1, vsstd);
        float32x4_t out2 = vmlaq_f32(vbias, submean2, vsstd);
        float32x4_t out3 = vmlaq_f32(vbias, submean3, vsstd);
        vst1q_f32(out_p, out0);
        vst1q_f32(out_p + 4, out1);
        vst1q_f32(out_p + 8, out2);
        vst1q_f32(out_p + 12, out3);
        in_p += 16;
        out_p += 16;
      }
      for (int k = 0; k < remain - 3; k += 4) {
        float32x4_t in0 = vld1q_f32(in_p);
        in_p += 4;
        float32x4_t submean0 = vsubq_f32(in0, vmean);
        float32x4_t out0 = vmlaq_f32(vbias, submean0, vsstd);
        vst1q_f32(out_p, out0);
        out_p += 4;
      }
      for (int k = 0; k < remain % 4; k++) {
        *out_p = (*in_p - mean_val) * sstd_val + bias_val;
        in_p++;
        out_p++;
      }
    }
  }
  LITE_PARALLEL_END()
  delete[] std_vec;
}

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_KERNEL(group_norm,
                     kARM,
                     kFloat,
                     kNCHW,
                     paddle::lite::kernels::arm::GroupNormCompute,
                     def)
    .BindInput("X", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Y", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Mean", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("Variance", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
