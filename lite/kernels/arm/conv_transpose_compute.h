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

#pragma once
#include <string>
#include <vector>
#include "lite/backends/arm/math/funcs.h"
#include "lite/core/kernel.h"
#include "lite/operators/conv_transpose_op.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace arm {
template <PrecisionType Ptype, PrecisionType Otype>
class Conv2DTransposeCompute : public KernelLite<TARGET(kARM), Ptype> {
 public:
  using param_t = operators::ConvParam;

  void PrepareForRun() override;

  void Run() override;

  virtual void ReInitWhenNeeded() {
    auto& param = this->template Param<param_t>();
    auto x_dims = param.x->dims();
    if (last_shape_ == x_dims) {
      return;
    }
    auto w_dims = param.filter->dims();
    auto o_dims = param.output->dims();
    int chin = x_dims[1];
    int hin = x_dims[2];
    int win = x_dims[3];
    int chout = o_dims[1];
    int kw = w_dims[3];
    int kh = w_dims[2];
    int group = param.groups;
    /* deconv weights layout: chin * chout * kh * kw*/
    int m = chout * kw * kh / group;
    int n = hin * win;
    int k = chin / group;
    workspace_size_ = group * m * n;
    last_shape_ = x_dims;
  }

  ~Conv2DTransposeCompute() = default;

#ifdef LITE_WITH_PROFILE
  virtual void SetProfileRuntimeKernelInfo(
      paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
  }
  std::string kernel_func_name_{"ConvTranspose"};
#define PROFILE_INFO(dtype1, dtype2)                                        \
  template <>                                                               \
  void Conv2DTransposeCompute<PRECISION(dtype1), PRECISION(dtype2)>::       \
      SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) { \
    ch->kernel_func_name = kernel_func_name_;                               \
  }

#define KERNEL_FUNC_NAME(kernel_func_name) kernel_func_name_ = kernel_func_name;

#else
#define PROFILE_INFO(dtype1, dtype2)
#define KERNEL_FUNC_NAME(kernel_func_name)
#endif

 protected:
  int workspace_size_{0};
  bool depthwise_{false};
  bool flag_trans_bias_{false};
  bool flag_trans_weight_{false};
  std::vector<float> w_scale_;
  DDim last_shape_;
  Tensor bias_;
  Tensor weights_;
};

}  // namespace arm
}  // namespace kernels
}  // namespace lite
}  // namespace paddle
