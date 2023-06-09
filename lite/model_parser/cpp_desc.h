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

#pragma once

#ifdef LITE_ON_FLATBUFFERS_DESC_VIEW
#include "lite/model_parser/flatbuffers/block_desc.h"
#include "lite/model_parser/flatbuffers/op_desc.h"
#include "lite/model_parser/flatbuffers/program_desc.h"
#include "lite/model_parser/flatbuffers/var_desc.h"
namespace paddle {
namespace lite {
namespace cpp {
using ProgramDesc = fbs::ProgramDescView;
using BlockDesc = fbs::BlockDescView;
using OpDesc = fbs::OpDescView;
using VarDesc = fbs::VarDescView;
using OpDescWrite = fbs::OpDesc;
}
}
}  // namespace paddle
#else
#include "lite/core/model/general/block_desc.h"
#include "lite/core/model/general/op_desc.h"
#include "lite/core/model/general/program_desc.h"
#include "lite/core/model/general/var_desc.h"
namespace paddle {
namespace lite {
namespace cpp {
using ProgramDesc = general::ProgramDesc;
using BlockDesc = general::BlockDesc;
using OpDesc = general::OpDesc;
using VarDesc = general::VarDesc;
using OpDescWrite = general::OpDesc;
}
}
}  // namespace paddle
#endif  // LITE_ON_FLATBUFFERS_DESC_VIEW
