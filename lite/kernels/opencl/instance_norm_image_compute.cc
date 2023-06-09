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

#include <memory>
#include <string>
#include "lite/backends/opencl/cl_half.h"
#include "lite/backends/opencl/cl_image_converter.h"
#include "lite/backends/opencl/cl_include.h"
#include "lite/core/kernel.h"
#include "lite/core/op_registry.h"
#include "lite/kernels/opencl/image_helper.h"
#include "lite/operators/op_params.h"
#include "lite/utils/log/logging.h"
#include "lite/utils/replace_stl/stream.h"
#ifdef LITE_WITH_PROFILE
#include "lite/core/profile/profiler.h"
#endif
#include "lite/backends/opencl/cl_utility.h"

namespace paddle {
namespace lite {
namespace kernels {
namespace opencl {
class InstanceNormImageCompute : public KernelLite<TARGET(kOpenCL),
                                                   PRECISION(kFP16),
                                                   DATALAYOUT(kImageDefault)> {
 public:
  using param_t = operators::InstanceNormParam;

  std::string doc() const override {
    return "InstanceNorm using cl::Image2D(ImageDefault/RGBA), kFP16";
  }

#if 1
  void PrepareForRun() override {
    instance_norm_param_ = param_.get_mutable<param_t>();
    auto out_h = instance_norm_param_->out->dims()[2];

    // TODO(ysh329): add instance_norm + relu pass
    // std::string build_options_ += "-DRELU";
    const bool enable_fp16 =
        CLRuntime::Global()->get_precision() == lite_api::CL_PRECISION_FP16;
    if (enable_fp16) {
      build_options_ += " -DCL_DTYPE_half -DCL_DTYPE_FLOAT_FORCE ";
    }
    if (out_h == 128) {
      build_options_ += " -DLOCAL_MEM_128";
    } else if (out_h == 64) {
      build_options_ += " -DLOCAL_MEM_64";
    }
    if (instance_norm_param_->activation_type == "relu") {
      build_options_ += " -DRELU";
    }
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/instance_norm_kernel.cl",
                                    build_options_,
                                    time_stamp_);
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    kernel_ = context.cl_context()->GetKernel(kernel_key.str());

    auto& out_dims = instance_norm_param_->out->dims();
    int batch = out_dims[0];
    int channel = out_dims[1];
    int cgroup = (channel + 3) / 4;
    int cround = cgroup * 4;

    const float* scale_data = instance_norm_param_->scale->data<float>();
    const float* bias_data = instance_norm_param_->bias->data<float>();

    std::vector<float> scale_img(cround * batch);
    std::vector<float> bias_img(cround * batch);

    std::vector<half_t> scale_img_h(cround * batch);
    std::vector<half_t> bias_img_h(cround * batch);

    DDim scale_img_size{{ cgroup, batch }};

    if (enable_fp16) {
      for (int i = 0; i < channel; ++i) {
        scale_img_h[i] = Float2Half(scale_data[i]);
        bias_img_h[i] = Float2Half(bias_data[i]);
      }

      for (int i = 1; i < batch; ++i) {
        memcpy(scale_img_h.data() + i * cround,
               scale_img_h.data(),
               cround * sizeof(half_t));
        memcpy(bias_img_h.data() + i * cround,
               bias_img_h.data(),
               cround * sizeof(half_t));
      }
      MUTABLE_DATA_GPU(&scale_image_,
                       scale_img_size[0],
                       scale_img_size[1],
                       scale_img_h.data());
      MUTABLE_DATA_GPU(&bias_image_,
                       scale_img_size[0],
                       scale_img_size[1],
                       bias_img_h.data());
    } else {
      for (int i = 0; i < channel; ++i) {
        scale_img[i] = scale_data[i];
        bias_img[i] = bias_data[i];
      }

      for (int i = 1; i < batch; ++i) {
        memcpy(scale_img.data() + i * cround,
               scale_img.data(),
               cround * sizeof(float));
        memcpy(bias_img.data() + i * cround,
               bias_img.data(),
               cround * sizeof(float));
      }
      MUTABLE_DATA_GPU(&scale_image_,
                       scale_img_size[0],
                       scale_img_size[1],
                       scale_img.data());
      MUTABLE_DATA_GPU(
          &bias_image_, scale_img_size[0], scale_img_size[1], bias_img.data());
    }
  }

  void ReInitWhenNeeded() override {
    instance_norm_param_ = param_.get_mutable<param_t>();
    auto x_dims = instance_norm_param_->x->dims();

    if ((!first_epoch_for_reinit_ && x_dims != last_x_dims_) ||
        first_epoch_for_reinit_) {
      last_x_dims_ = x_dims;
      first_epoch_for_reinit_ = false;

      // compute global/local work size
      auto device_info = CLRuntime::Global()->GetDeviceInfo();
      int max_work_item_size1 = device_info["CL_DEVICE_MAX_WORK_ITEM_SIZES_1"];
      int lws0 = 1;
      int lws1 = std::min(max_work_item_size1,
                          std::min(256, static_cast<int>(x_dims[3])));
      int lws2 = 1;
      gws_ = cl::NDRange{
          static_cast<cl::size_type>(x_dims[0] * ((x_dims[1] + 3) / 4)),
          static_cast<cl::size_type>(lws1),
          static_cast<cl::size_type>(lws2)};
      lws_ = cl::NDRange{static_cast<cl::size_type>(lws0),
                         static_cast<cl::size_type>(lws1),
                         static_cast<cl::size_type>(lws2)};
    }
  }

  void Run() override {
    auto& context = ctx_->As<OpenCLContext>();

    auto* x = instance_norm_param_->x;
    auto* out = instance_norm_param_->out;
    auto& out_dims = out->dims();

    const int out_c_group = (out_dims[1] + 3) / 4;
    const int out_h = out_dims[2];
    const int out_w = out_dims[3];

    float epsilon = instance_norm_param_->epsilon;

#ifdef LITE_WITH_LOG
    VLOG(4) << "global_work_size:" << static_cast<int>(gws_[0]) << " "
            << static_cast<int>(gws_[1]) << " " << static_cast<int>(gws_[2]);
    VLOG(4) << "local_work_size:" << static_cast<int>(lws_[0]) << " "
            << static_cast<int>(lws_[1]) << " " << static_cast<int>(lws_[2]);
    VLOG(4) << "out_w:" << out_w;
    VLOG(4) << "out_h:" << out_h;
    VLOG(4) << "out_c_group:" << out_c_group;
    VLOG(4) << "lws1:" << lws_[1];
    VLOG(4) << "lws2:" << lws_[2];
    VLOG(4) << "epsilon:" << epsilon;
#endif

    auto out_image_shape = InitImageDimInfoWith(out_dims);
    auto* x_img = GET_DATA_GPU(x);
    auto* out_img = MUTABLE_DATA_GPU(
        out, out_image_shape["width"], out_image_shape["height"], nullptr);
    auto* scale_img = GET_DATA_GPU(&scale_image_);
    auto* bias_img = GET_DATA_GPU(&bias_image_);

    cl_int status = kernel_.setArg(0, out_w);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(1, out_h);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(2, out_c_group);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(3, static_cast<int>(lws_[1]));
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(4, static_cast<int>(lws_[2]));
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(5, epsilon);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(6, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(7, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(8, *scale_img);
    CL_CHECK_FATAL(status);
    status = kernel_.setArg(9, *bias_img);
    CL_CHECK_FATAL(status);

    status = EnqueueNDRangeKernel(
        context, kernel_, cl::NullRange, gws_, lws_, nullptr, event_);

    CL_CHECK_FATAL(status);
  }

#else  // paddle version
  void PrepareForRun() override {
    instance_norm_param_ = param_.get_mutable<param_t>();
    auto channel = instance_norm_param_->scale->dims()[0];
    auto batch = instance_norm_param_->x->dims()[0];
    int64_t cgroup = (channel + 3) / 4;
    int64_t cround = cgroup * 4;
    std::vector<half_t> scale_img(cround * batch);
    std::vector<half_t> bias_img(cround * batch);
    const float* scale_data = instance_norm_param_->scale->data<float>();
    const float* bias_data = instance_norm_param_->bias->data<float>();
    //! init scale_img bias_img data
    for (int i = 0; i < channel; ++i) {
      scale_img[i] = Float2Half(scale_data[i]);
      bias_img[i] = Float2Half(bias_data[i]);
    }
    for (int i = channel; i < cround; ++i) {
      scale_img[i] = Float2Half(0.f);
      bias_img[i] = Float2Half(0.f);
    }
    for (int i = 1; i < batch; ++i) {
      memcpy(scale_img.data() + i * cround,
             scale_img.data(),
             cround * sizeof(half_t));
      memcpy(bias_img.data() + i * cround,
             bias_img.data(),
             cround * sizeof(half_t));
    }
    DDim scale_img_size{{cgroup, batch}};
    scale_image_.mutable_data<half_t, cl::Image2D>(
        scale_img_size[0], scale_img_size[1], scale_img.data());
    bias_image_.mutable_data<half_t, cl::Image2D>(
        scale_img_size[0], scale_img_size[1], bias_img.data());
    auto& context = ctx_->As<OpenCLContext>();
    context.cl_context()->AddKernel(kernel_func_name_,
                                    "image/instance_norm_kernel.cl",
                                    build_options_,
                                    time_stamp_);
    VLOG(1) << "kernel_func_name_:" << kernel_func_name_;
  }

  void Run() override {
    auto& context = ctx_->As<OpenCLContext>();
    CHECK(context.cl_context() != nullptr);
    auto* x = instance_norm_param_->x;
    auto* out = instance_norm_param_->out;
    auto in_dims = x->dims();

    int batch = in_dims[0];
    int channel = in_dims[1];
    int in_h = in_dims[2];
    int in_w = in_dims[3];

#ifdef LITE_WITH_LOG
    VLOG(4) << "x->target():" << TargetToStr(x->target());
    VLOG(4) << "out->target():" << TargetToStr(out->target());
    VLOG(4) << "x->dims():" << in_dims;
#endif

    auto out_image_shape = InitImageDimInfoWith(in_dims);
    auto* x_img = x->data<half_t, cl::Image2D>();
    auto* out_img = out->mutable_data<half_t, cl::Image2D>(
        out_image_shape["width"], out_image_shape["height"]);

#ifdef LITE_WITH_LOG
    VLOG(4) << "out_image_shape[w,h]: " << out_image_shape["width"] << " "
            << out_image_shape["height"];

    VLOG(4) << "in_h: " << in_h << ", in_w: " << in_w;
#endif

    int threads = 512;
    int group_size_x = (channel + 3) / 4;
    int group_size_y = batch;
    auto local_work_size = cl::NDRange{static_cast<cl::size_type>(threads),
                                       static_cast<cl::size_type>(1),
                                       static_cast<cl::size_type>(1)};
    auto global_work_size =
        cl::NDRange{static_cast<cl::size_type>(group_size_x * threads),
                    static_cast<cl::size_type>(group_size_y),
                    static_cast<cl::size_type>(1)};

#ifdef LITE_WITH_LOG
    VLOG(4) << "local_work_size:[2D]:" << local_work_size[0] << " "
            << local_work_size[1] << " " << local_work_size[2];
    VLOG(4) << "global_work_size:[2D]:" << global_work_size[0] << " "
            << global_work_size[1] << " " << global_work_size[2];
#endif

    STL::stringstream kernel_key;
    kernel_key << kernel_func_name_ << build_options_ << time_stamp_;
    auto kernel = context.cl_context()->GetKernel(kernel_key.str());
    auto* scale_img = scale_image_.data<half_t, cl::Image2D>();
    auto* bias_img = bias_image_.data<half_t, cl::Image2D>();
    float epsilon = instance_norm_param_->epsilon;

    cl_int status = kernel.setArg(arg_idx++, *x_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, *out_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, *scale_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, *bias_img);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, epsilon);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, in_h);
    CL_CHECK_FATAL(status);
    status = kernel.setArg(arg_idx++, in_w);
    CL_CHECK_FATAL(status);

    status = EnqueueNDRangeKernel(context,
                                  kernel,
                                  cl::NullRange,
                                  global_work_size,
                                  local_work_size,
                                  nullptr,
                                  event_);
    CL_CHECK_FATAL(status);
  }
#endif

#ifdef LITE_WITH_PROFILE
  void SetProfileRuntimeKernelInfo(paddle::lite::profile::OpCharacter* ch) {
    ch->kernel_func_name = kernel_func_name_;
    ch->cl_event =
        event_;  // `event_` defined in `kernel.h`, valid after kernel::Run
  }
#endif

 protected:
  param_t* instance_norm_param_{nullptr};
  bool first_epoch_for_reinit_{true};
  DDim last_x_dims_;
  std::string kernel_func_name_{"instance_norm"};
  std::string build_options_{""};
  std::string time_stamp_{GetTimeStamp()};
  cl::Kernel kernel_;
  cl::NDRange gws_, lws_;
  Tensor scale_image_;
  Tensor bias_image_;
};

}  // namespace opencl
}  // namespace kernels
}  // namespace lite
}  // namespace paddle

namespace ocl = paddle::lite::kernels::opencl;
REGISTER_LITE_KERNEL(instance_norm,
                     kOpenCL,
                     kFP16,
                     kImageDefault,
                     ocl::InstanceNormImageCompute,
                     ImageDefault)
    .BindInput("X",
               {LiteType::GetTensorTy(TARGET(kOpenCL),
                                      PRECISION(kFP16),
                                      DATALAYOUT(kImageDefault))})
    .BindOutput("Y",
                {LiteType::GetTensorTy(TARGET(kOpenCL),
                                       PRECISION(kFP16),
                                       DATALAYOUT(kImageDefault))})
    .BindInput("Scale", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindInput("Bias", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedMean", {LiteType::GetTensorTy(TARGET(kARM))})
    .BindOutput("SavedVariance", {LiteType::GetTensorTy(TARGET(kARM))})
    .Finalize();
