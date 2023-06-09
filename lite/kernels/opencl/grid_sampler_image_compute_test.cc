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
#include <string>
#include "lite/backends/opencl/target_wrapper.h"
#include "lite/core/op_registry.h"
#include "lite/core/tensor.h"
#include "lite/kernels/opencl/test_helper.h"

#define FP16_RELATIVE_DIFF (5e-2)
#define FP16_ABS_DIFF (5e-2)
#define FP32_RELATIVE_DIFF (1e-3)
#define FP32_ABS_DIFF (5e-4)

namespace paddle {
namespace lite {

void gird_sampler_ref(const float* din,
                      const DDim& in_dims,
                      const float* grid,
                      float* output,
                      const std::string& mode,
                      const std::string& padding_mode,
                      bool align_corners) {
  int num = in_dims[0];
  int channel = in_dims[1];
  int height = in_dims[2];
  int width = in_dims[3];
  int spatial_size = height * width;

  auto inbound = [](int x, int y, float x_max, float y_max) {
    if (x < 0 || x > x_max || y < 0 || y > y_max) {
      return false;
    }
    return true;
  };

  for (int n = 0; n < num; ++n) {
    const float* x_n = din + n * channel * height * width;
    float* out_n = output + n * channel * height * width;
    const float* grid_n = grid + n * height * width * 2;
    for (int c = 0; c < channel; ++c) {
      const float* x_c = x_n + c * spatial_size;
      float* out_c = out_n + c * spatial_size;
      for (int s = 0; s < spatial_size; ++s) {
        float x = grid_n[s * 2];
        float y = grid_n[s * 2 + 1];
        float xwf = (x + 1.f) * 0.5 * (width - 1);
        float ynf = (y + 1.f) * 0.5 * (height - 1);
        int xw = floor(xwf);
        int xe = xw + 1;
        int yn = floor(ynf);
        int ys = yn + 1;

        float dw = xwf - xw;
        float de = xe - xwf;
        float dn = ynf - yn;
        float ds = ys - ynf;

        float wn = inbound(xw,
                           yn,
                           static_cast<float>(width - 1),
                           static_cast<float>(height - 1))
                       ? x_c[yn * width + xw]
                       : 0.f;
        float en = inbound(xe,
                           yn,
                           static_cast<float>(width - 1),
                           static_cast<float>(height - 1))
                       ? x_c[yn * width + xe]
                       : 0.f;
        float ws = inbound(xw,
                           ys,
                           static_cast<float>(width - 1),
                           static_cast<float>(height - 1))
                       ? x_c[ys * width + xw]
                       : 0.f;
        float es = inbound(xe,
                           ys,
                           static_cast<float>(width - 1),
                           static_cast<float>(height - 1))
                       ? x_c[ys * width + xe]
                       : 0.f;

        out_c[s] = wn * de * ds + en * dw * ds + ws * de * dn + es * dw * dn;
      }
    }
  }
}
// #define GRID_FP16_LOOP_TEST
// #define GRID_FP16_PRINT_RESULT
void test_precision(const lite_api::CLPrecisionType p) {
#ifdef GRID_FP16_LOOP_TEST
  for (bool align_corners : {true, false}) {
    for (const std::string& mode : {"bilinear", "nearest"}) {
      for (const std::string& padding_mode :
           {"zeros", "reflection", "border"}) {
        for (int n = 1; n <= 100; n += 33) {
          for (auto c : {1, 3, 8, 23, 32}) {
            for (int h = 12; h <= 100; h += 13) {
              for (int w = 12; w <= 100; w += 25) {
#else
  const std::string& mode = "bilinear";
  const std::string& padding_mode = "zeros";
  bool align_corners = true;
  const int n = 1;
  const int c = 2;
  const int h = 5;
  const int w = 4;
#endif  // GRID_FP16_LOOP_TEST
                LOG(INFO) << "======== input shape[n,c,h,w]:" << n << " " << c
                          << " " << h << " " << w << " , mode:" << mode
                          << ", padding_mode:" << padding_mode
                          << ", align_corners:" << align_corners << " ========";
                CLRuntime::Global()->set_precision(p);
                const bool fp16_flag =
                    (p == lite_api::CLPrecisionType::CL_PRECISION_FP16);
                auto kernels =
                    KernelRegistry::Global().Create("grid_sampler",
                                                    TARGET(kOpenCL),
                                                    PRECISION(kFP16),
                                                    DATALAYOUT(kImageDefault));
                ASSERT_FALSE(kernels.empty());
                auto kernel = std::move(kernels.front());
                LOG(INFO) << "get kernel:" << kernel->doc();

                lite::Tensor x, grid, out;
                operators::GridSamplerParam param;
                param.x = &x;
                param.grid = &grid;
                param.out = &out;
                param.mode = mode;
                param.padding_mode = padding_mode;
                param.align_corners = align_corners;

                std::unique_ptr<KernelContext> context(new KernelContext);
                context->As<OpenCLContext>().InitOnce();

                kernel->SetParam(param);
                std::unique_ptr<KernelContext> grid_context(new KernelContext);
                context->As<OpenCLContext>().CopySharedTo(
                    &(grid_context->As<OpenCLContext>()));
                kernel->SetContext(std::move(grid_context));

                const DDim in_dim =
                    DDim(std::vector<DDim::value_type>{n, c, h, w});
                const DDim grid_dim =
                    DDim(std::vector<DDim::value_type>{n, h, w, 2});
                const DDim out_dim =
                    DDim(std::vector<DDim::value_type>{n, c, h, w});
                x.Resize(in_dim);
                grid.Resize(grid_dim);
                out.Resize(out_dim);

                std::default_random_engine engine;
                std::uniform_real_distribution<float> dist(-1, 1);
                int sum = n * c * h * w;
                int sum2 = n * h * w * 2;
                std::vector<float> input_v(sum);
                std::vector<float> grid_v(sum2);
                std::vector<float> output_v(out_dim.production());
                for (auto& i : input_v) {
                  i = dist(engine);
                }
                for (auto& i : grid_v) {
                  i = dist(engine);
                }

                LOG(INFO) << "prepare input";
                CLImageConverterDefault* default_converter =
                    new CLImageConverterDefault();
                DDim x_image_shape =
                    default_converter->InitImageDimInfoWith(in_dim);
                LOG(INFO) << "x_image_shape = " << x_image_shape[0] << " "
                          << x_image_shape[1];
                const size_t dtype_size =
                    fp16_flag ? sizeof(half_t) : sizeof(float);
                std::vector<char> x_image_data(x_image_shape.production() * 4 *
                                               dtype_size);  // 4 : RGBA
                default_converter->NCHWToImage(
                    input_v.data(), x_image_data.data(), in_dim);
                MUTABLE_DATA_GPU(&x,
                                 x_image_shape[0],
                                 x_image_shape[1],
                                 x_image_data.data());

                DDim grid_image_shape =
                    default_converter->InitImageDimInfoWith(grid_dim);
                LOG(INFO) << "grid_image_shape = " << grid_image_shape[0] << " "
                          << grid_image_shape[1];
                std::vector<char> grid_image_data(
                    grid_image_shape.production() * 4 *
                    dtype_size);  // 4 : RGBA
                default_converter->NCHWToImage(
                    grid_v.data(), grid_image_data.data(), grid_dim);
                MUTABLE_DATA_GPU(&grid,
                                 grid_image_shape[0],
                                 grid_image_shape[1],
                                 grid_image_data.data());

                DDim out_image_shape =
                    default_converter->InitImageDimInfoWith(out_dim);
                LOG(INFO) << "out_image_shape = " << out_image_shape[0] << " "
                          << out_image_shape[1];
                auto* out_image = MUTABLE_DATA_GPU(
                    &out, out_image_shape[0], out_image_shape[1], nullptr);

                kernel->Launch();

                CLRuntime::Global()->command_queue().finish();

                std::vector<float> out_ref(out_dim.production());
                auto* out_ref_data = out_ref.data();
                gird_sampler_ref(input_v.data(),
                                 in_dim,
                                 grid_v.data(),
                                 out_ref_data,
                                 mode,
                                 padding_mode,
                                 align_corners);

                const size_t cl_image2d_row_pitch{0};
                const size_t cl_image2d_slice_pitch{0};
                std::vector<char> out_image_data(out_image_shape.production() *
                                                 4 * dtype_size);  // 4 : RGBA
                TargetWrapperCL::ImgcpySync(out_image_data.data(),
                                            out_image,
                                            out_image_shape[0],
                                            out_image_shape[1],
                                            cl_image2d_row_pitch,
                                            cl_image2d_slice_pitch,
                                            IoDirection::DtoH);
                default_converter->ImageToNCHW(out_image_data.data(),
                                               output_v.data(),
                                               out_image_shape,
                                               out_dim);
// result
#ifdef GRID_FP16_PRINT_RESULT
                LOG(INFO) << "---- print kernel result (input -> output) ----";
                for (int eidx = 0; eidx < in_dim.production(); ++eidx) {
                  std::cout << input_v[eidx] << " -> " << out_data[eidx]
                            << "\n";
                }
#endif  // GRID_FP16_PRINT_RESULT
                uint32_t diff_cnt = 0;
                auto relative_diff_thres =
                    fp16_flag ? FP16_RELATIVE_DIFF : FP32_RELATIVE_DIFF;
                auto abs_diff_thres = fp16_flag ? FP16_ABS_DIFF : FP32_ABS_DIFF;
                for (int i = 0; i < out_dim.production(); i++) {
                  auto abs_diff = abs(output_v[i] - out_ref_data[i]);
                  auto relative_diff =
                      COMPUTE_RELATIVE_DIFF(output_v[i], out_ref_data[i]);
                  EXPECT_FALSE(relative_diff > relative_diff_thres &&
                               abs_diff > abs_diff_thres);
                  if ((relative_diff > relative_diff_thres) &&
                      (abs_diff > abs_diff_thres)) {
                    LOG(WARNING) << i << ": \t out_ins: " << output_v[i]
                                 << "\t out_ref: " << out_ref_data[i];
                    diff_cnt++;
                  }
                }
                if (diff_cnt != 0) {
                  LOG(FATAL) << "Err num " << diff_cnt << "/"
                             << out_dim.production();
                }
#ifdef GRID_FP16_LOOP_TEST
              }  // w
            }    // h
          }      // c
        }        // n
      }          // padding_mode
    }            // mode
  }              // align_corners
#else
// nothing to do.
#endif
}

TEST(grid_sampler, compute_basic) {
  for (auto p : {lite_api::CLPrecisionType::CL_PRECISION_FP32,
                 lite_api::CLPrecisionType::CL_PRECISION_FP16}) {
    test_precision(p);
  }
}

}  // namespace lite
}  // namespace paddle

USE_LITE_KERNEL(grid_sampler, kOpenCL, kFP16, kImageDefault, ImageDefault);
