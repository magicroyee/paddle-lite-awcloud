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

#include <gtest/gtest.h>
#include <memory>
#include <random>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/test_helper.h"

#define FP16_MAX_DIFF (5e-1)

namespace paddle {
namespace lite {

void sin(const float* input_data, const DDim& in_dim, float* output_data) {
  for (int i = 0; i < in_dim.production(); i++) {
    output_data[i] = std::sin(input_data[i]);
  }
}

void cos(const float* input_data, const DDim& in_dim, float* output_data) {
  for (int i = 0; i < in_dim.production(); i++) {
    output_data[i] = std::cos(input_data[i]);
  }
}

void tan(const float* input_data, const DDim& in_dim, float* output_data) {
  for (int i = 0; i < in_dim.production(); i++) {
    output_data[i] = std::tan(input_data[i]);
  }
}

void atan(const float* input_data, const DDim& in_dim, float* output_data) {
  for (int i = 0; i < in_dim.production(); i++) {
    output_data[i] = std::atan(input_data[i]);
  }
}

void asin(const float* input_data, const DDim& in_dim, float* output_data) {
  for (int i = 0; i < in_dim.production(); i++) {
    output_data[i] = std::asin(input_data[i]);
  }
}

void acos(const float* input_data, const DDim& in_dim, float* output_data) {
  for (int i = 0; i < in_dim.production(); i++) {
    output_data[i] = std::acos(input_data[i]);
  }
}

TEST(trigonometrics_image2d_fp16, compute) {
  std::vector<std::string> trigonometrics{
      "sin", "cos", "tan", "atan", "asin", "acos"};
  for (size_t i = 0; i < trigonometrics.size(); i++) {
    auto trigonometric = trigonometrics[i];

    LOG(INFO) << "trigonometric:   " << trigonometric;
    auto kernels = KernelRegistry::Global().Create(trigonometric,
                                                   TARGET(kOpenCL),
                                                   PRECISION(kFP16),
                                                   DATALAYOUT(kImageDefault));
    ASSERT_FALSE(kernels.empty());

    auto kernel = std::move(kernels.front());

    LOG(INFO) << "Get kernel:" << kernel->doc();

    lite::Tensor x, out;
    operators::TrigonometricParam param;
    param.X = &x;
    param.Out = &out;

    std::unique_ptr<KernelContext> context(new KernelContext);
    context->As<OpenCLContext>().InitOnce();

    kernel->SetParam(param);
    std::unique_ptr<KernelContext> trigonometric_context(new KernelContext);
    context->As<OpenCLContext>().CopySharedTo(
        &(trigonometric_context->As<OpenCLContext>()));
    kernel->SetContext(std::move(trigonometric_context));

    const DDim in_dim = DDim(std::vector<DDim::value_type>{4, 11, 107, 107});
    const DDim out_dim = DDim(std::vector<DDim::value_type>{4, 11, 107, 107});
    x.Resize(in_dim);
    out.Resize(out_dim);

    std::default_random_engine engine;
    std::uniform_real_distribution<float> dist(-1, 1);
    std::vector<float> input_v(4 * 11 * 107 * 107);
    for (auto& i : input_v) {
      i = dist(engine);
    }

    LOG(INFO) << "prepare input";
    CLImageConverterDefault* default_converter = new CLImageConverterDefault();
    DDim image_shape = default_converter->InitImageDimInfoWith(in_dim);
    LOG(INFO) << "image_shape = " << image_shape[0] << " " << image_shape[1];
    std::vector<half_t> x_image_data(image_shape.production() * 4);  // 4 : RGBA
    default_converter->NCHWToImage(input_v.data(), x_image_data.data(), in_dim);
    auto* x_image = x.mutable_data<half_t, cl::Image2D>(
        image_shape[0], image_shape[1], x_image_data.data());
    LOG(INFO) << "x_image:" << x_image;

    auto* out_image =
        out.mutable_data<half_t, cl::Image2D>(image_shape[0], image_shape[1]);
    LOG(INFO) << "out_image:" << out_image;
    kernel->Launch();

    CLRuntime::Global()->command_queue().finish();

    std::unique_ptr<float[]> out_ref(new float[out_dim.production()]);

    if (trigonometric == "sin") {
      sin(input_v.data(), in_dim, out_ref.get());
    } else if (trigonometric == "cos") {
      cos(input_v.data(), in_dim, out_ref.get());
    } else if (trigonometric == "tan") {
      tan(input_v.data(), in_dim, out_ref.get());
    } else if (trigonometric == "atan") {
      atan(input_v.data(), in_dim, out_ref.get());
    } else if (trigonometric == "asin") {
      asin(input_v.data(), in_dim, out_ref.get());
    } else if (trigonometric == "acos") {
      acos(input_v.data(), in_dim, out_ref.get());
    }

    const size_t cl_image2d_row_pitch{0};
    const size_t cl_image2d_slice_pitch{0};
    half_t* out_image_data = new half_t[image_shape.production() * 4];
    TargetWrapperCL::ImgcpySync(out_image_data,
                                out_image,
                                image_shape[0],
                                image_shape[1],
                                cl_image2d_row_pitch,
                                cl_image2d_slice_pitch,
                                IoDirection::DtoH);
    float* out_data = new float[image_shape.production() * 4];
    default_converter->ImageToNCHW(
        out_image_data, out_data, image_shape, out_dim);

    for (int i = 0; i < out_dim.production(); i++) {
      auto abs_diff = abs(out_data[i] - out_ref[i]);
      auto relative_diff = COMPUTE_RELATIVE_DIFF(out_data[i], out_ref[i]);
      EXPECT_EQ((relative_diff <= FP16_MAX_DIFF) || (abs_diff <= FP16_MAX_DIFF),
                true);
      if ((relative_diff > FP16_MAX_DIFF) && (abs_diff > FP16_MAX_DIFF)) {
        LOG(ERROR) << "error idx:" << i << " out_data[" << i
                   << "]:" << out_data[i] << " out_ref[" << i
                   << "]:" << out_ref[i] << " abs_diff:" << abs_diff
                   << " relative_diff:" << relative_diff
                   << " FP16_MAX_DIFF:" << FP16_MAX_DIFF;
      }
    }
  }
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(sin, kOpenCL, kFP16, kImageDefault, image2d);
USE_LITE_KERNEL(cos, kOpenCL, kFP16, kImageDefault, image2d);
USE_LITE_KERNEL(tan, kOpenCL, kFP16, kImageDefault, image2d);
USE_LITE_KERNEL(atan, kOpenCL, kFP16, kImageDefault, image2d);
USE_LITE_KERNEL(asin, kOpenCL, kFP16, kImageDefault, image2d);
USE_LITE_KERNEL(acos, kOpenCL, kFP16, kImageDefault, image2d);
