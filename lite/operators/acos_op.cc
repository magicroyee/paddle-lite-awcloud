// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include "lite/operators/acos_op.h"
#include "lite/core/op_registry.h"

namespace paddle {
namespace lite {
namespace operators {

bool AcosOpLite::CheckShape() const {
  CHECK_OR_FALSE(param_.X);
  CHECK_OR_FALSE(param_.Out);
  return true;
}

bool AcosOpLite::InferShape() {
  lite::DDim x_dims = param_.X->dims();
  param_.Out->Resize(x_dims);

  return true;
}

bool AcosOpLite::AttachImpl(const cpp::OpDesc &op_desc, lite::Scope *scope) {
  auto x = op_desc.Input("X").front();
  auto out = op_desc.Output("Out").front();

  param_.X = scope->FindVar(x)->GetMutable<lite::Tensor>();
  param_.Out = scope->FindVar(out)->GetMutable<lite::Tensor>();

  return true;
}

}  // namespace operators
}  // namespace lite
}  // namespace paddle

REGISTER_LITE_OP(acos, paddle::lite::operators::AcosOpLite);
