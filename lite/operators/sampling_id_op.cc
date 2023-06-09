// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/sampling_id_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool SamplingIdOp::CheckShape() const {
  CHECK(param_.x);
  CHECK(param_.out);
  return true;
}

bool SamplingIdOp::InferShapeImpl() const {
  auto x_dims = param_.x->dims();
  CHECK_EQ(x_dims.size(), 2UL);
  param_.out->Resize(DDim{{x_dims[0]}});
  param_.out->set_lod(param_.x->lod());
  return true;
}

bool SamplingIdOp::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  param_.x = scope->FindTensor(op_desc.Input("X").front());
  param_.out = scope->FindMutableTensor(op_desc.Output("Out").front());

  param_.min = op_desc.GetAttr<float>("min");
  param_.max = op_desc.GetAttr<float>("max");
  param_.seed = op_desc.GetAttr<int>("seed");
  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(sampling_id, paddle::lite::operators::SamplingIdOp);
