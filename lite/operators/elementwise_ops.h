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
#include "lite/core/op_lite.h"

namespace paddle {
namespace lite {
namespace operators {

class ElementwiseOp : public OpLite {
 public:
  explicit ElementwiseOp(const std::string& op_type) : OpLite(op_type) {}

  bool CheckShape() const override;

  bool InferShapeImpl() const override;

  bool InferShapeWithCache() const override { return true; }

  bool AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) override;

  void AttachKernel(KernelBase* kernel) override { kernel->SetParam(param_); }

  std::string DebugString() const override { return "elementwise_op"; }

#ifdef LITE_WITH_PROFILE
  void GetOpRuntimeInfo(paddle::lite::profile::OpCharacter* ch) {
    auto output_dims = param_.Out->dims();
    ch->input_shape = "X" + ch->DimToStr(param_.X->dims()) + "Y" +
                      ch->DimToStr(param_.Y->dims());
    ch->output_shape = ch->DimToStr(output_dims);
    ch->remark = "axis" + std::to_string(param_.axis);
    ch->macs = 1.0f * param_.Out->numel();
  }
#endif

 private:
  mutable operators::ElementwiseParam param_;
};

// #ifdef LITE_WITH_TRAIN
// class ElementwiseGradExplicitOp : public OpLite {
//  public:
//   explicit ElementwiseGradExplicitOp(const std::string& type) : OpLite(type)
//   {}

//   bool CheckShape() const override;

//   bool InferShapeImpl() const override;

//   bool AttachImpl(const cpp::OpDesc& opdesc, lite::Scope* scope) override;

//   void AttachKernel(KernelBase* kernel) override { kernel->SetParam(param_);
//   }

//   std::string DebugString() const override {
//     return "elementwise_grad_explicit_op";
//   }

//  private:
//   mutable operators::ElementwiseGradParam param_;
// };
// #endif

}  // namespace operators
}  // namespace lite
}  // namespace paddle
