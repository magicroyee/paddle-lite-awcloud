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

#include "operation/softmax.h"
#include "driver/mediatek_apu/converter/converter.h"
#include "utility/debug.h"
#include "utility/logging.h"

namespace nnadapter {
namespace mediatek_apu {

int ConvertSoftmax(Converter* converter, core::Operation* operation) {
  SOFTMAX_OPERATION_EXTRACT_INPUTS_OUTPUTS

  // Convert to Neuron operands and operations
  auto input_index = converter->GetMappedIndex(input_operand);
  if (input_index == INVALID_INDEX) {
    input_index = converter->ConvertOperand(input_operand);
  }
  auto beta_index = converter->AddFloat32ConstantOperand(1.0f);
  auto axis_index = converter->AddInt32ConstantOperand(axis);
  auto output_index = converter->ConvertOperand(output_operand);
  NNADAPTER_CHECK_EQ(
      converter->AddOperation(NEURON_SOFTMAX,
                              {input_index, beta_index, axis_index},
                              {output_index}),
      NEURON_NO_ERROR);
  return NNADAPTER_NO_ERROR;
}

}  // namespace mediatek_apu
}  // namespace nnadapter
