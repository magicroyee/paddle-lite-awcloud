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

#include <intelfpga.h>
#include "lite/kernels/intel_fpga/bridges/graph.h"
#include "lite/core/subgraph/subgraph_bridge_registry.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace intel_fpga {

int ElementwiseConverter(void *ctx, OpLite *op, KernelBase *kernel) {
    VLOG(4) << "Converting elementwise op for intelfpga.";
    CHECK(ctx != nullptr);
    CHECK(op != nullptr);
    auto graph = static_cast<Graph*>(ctx);

    std::string op_type = op->op_info()->Type();
    auto op_info = op->op_info();
    auto scope = op->scope();

    auto input_x_name = op_info->Input("X").front();
    auto input_y_name = op_info->Input("Y").front();
    auto output_name = op_info->Output("Out").front();
    auto x_tensor = scope->FindMutableTensor(input_x_name);
    auto y_tensor = scope->FindMutableTensor(input_y_name);
    auto out_tensor = scope->FindMutableTensor(output_name);
    VLOG(4) << "output_name:" << output_name;

    auto i_dims_x = x_tensor->dims();
    auto i_dims_y = y_tensor->dims();
    auto o_dims = out_tensor->dims();
    CHECK_EQ(i_dims_x[0], i_dims_y[0]);
    CHECK_EQ(i_dims_x[1], i_dims_y[1]);
    CHECK_EQ(i_dims_x[2], i_dims_y[2]);
    CHECK_EQ(i_dims_x[3], i_dims_y[3]);

    FpgaElementwiseParam* ele_param = new FpgaElementwiseParam();
    auto& add_param = ele_param->add_param;

    ele_param->input_x = x_tensor->mutable_data<int8_t>();
    ele_param->input_y = y_tensor->mutable_data<int8_t>();
    ele_param->output = out_tensor->mutable_data<int8_t>();

    std::vector<float> x_scales;
    std::vector<float> y_scales;
    auto x_scale_name = "X0_scale";
    if (op_info->HasAttr("forced_scale") &&
        op_info->HasAttr(input_x_name + "_forced_scale") &&
        op_info->HasAttr(input_y_name + "_forced_scale")) {
      x_scales.push_back(op_info->GetAttr<float>(input_x_name + "_forced_scale"));
      y_scales.push_back(op_info->GetAttr<float>(input_y_name + "_forced_scale"));
    } else {
      if (op_info->HasInputScale(x_scale_name, true)) {
        x_scales = op_info->GetInputScale(x_scale_name, true);
      } else {
        x_scales = {-1.0};
      }
      auto y_scale_name = "Y0_scale";
      if (op_info->HasInputScale(y_scale_name, true)) {
        y_scales = op_info->GetInputScale(y_scale_name, true);
      } else {
        y_scales = {-1.0};
      }
    }
    VLOG(4) << "X0_scale: " << x_scales[0];
    VLOG(4) << "Y0_scale: " << y_scales[0];
    auto out_scale_name = "Out0_scale";
    std::vector<float> out_scales;

    if (op_info->HasOutputScale(out_scale_name, true)) {
      out_scales = op_info->GetOutputScale(out_scale_name, true);
    } else {
      out_scales = {-1};
    }
    VLOG(4) << "Out0_scale: " << out_scales[0];

    int32_t axis = op_info->GetAttr<int32_t>("axis");
    if (axis != -1) {
      LOG(FATAL) << "Only support axis =-1 for elementwise op.";
    }
    auto act_type =
      op_info->HasAttr("act_type") ? op_info->GetAttr<std::string>("act_type") : "";
    VLOG(4) << "act_type: " << act_type;
    if (act_type != "" && act_type != "relu") {
      LOG(FATAL) << "Only support relu for activation for elementwise op.";
    }

    Node* node = new Node();
    node->name_ = output_name;
    node->op_type_ = INTELFPGA_ELE_ADD;
    node->is_output = graph->IsOutput(output_name);
    node->is_input = graph->IsInput(input_x_name) || graph->IsInput(input_y_name);
    node->node_param_ = dynamic_cast<NodeParam*>(ele_param);
    if (act_type == "relu") {
      ele_param->ac_type = INTELFPGA_ACT_RELU;
    }

    VLOG(4) << "input_x_name: " << input_x_name;
    if(graph->GetNodeByTensorName(input_x_name)) {
      node->parent_vec_.push_back(graph->GetNodeByTensorName(input_x_name));
      // Set input offset.
      int byte_offset =
          FpgaWord2ByteOffset(node->parent_vec_[0]->op_type_,
              FpgaGetOutputOffset(node->parent_vec_[0]));
      add_param.input1_offset = FpgaByte2WordOffset(node->op_type_, byte_offset);
    } else {
      node->parent_vec_.push_back(nullptr);
      VLOG(4) << "Malloc input_x.";
      device_output_config config = FpgaMemMalloc(node->op_type_,
          ele_param->d_x, i_dims_x[1], i_dims_x[2], i_dims_x[3]);
      add_param.input1_offset = config.output_offset;
    }

    VLOG(4) << "input_y_name: " << input_y_name;
    if(graph->GetNodeByTensorName(input_y_name)) {
      node->parent_vec_.push_back(graph->GetNodeByTensorName(input_y_name));
      // Set input offset.
      int byte_offset =
          FpgaWord2ByteOffset(node->parent_vec_[1]->op_type_,
              FpgaGetOutputOffset(node->parent_vec_[1]));
      add_param.input2_offset = FpgaByte2WordOffset(node->op_type_, byte_offset);
    } else {
      node->parent_vec_.push_back(nullptr);
      VLOG(4) << "Malloc input_y.";
      device_output_config config = FpgaMemMalloc(node->op_type_,
          ele_param->d_y, i_dims_y[1], i_dims_y[2], i_dims_y[3]);
      add_param.input2_offset = config.output_offset;
    }

    // Malloc output and set output offset.
    device_output_config config = FpgaMemMalloc(node->op_type_,
        ele_param->d_o, o_dims[1], o_dims[2], o_dims[3]);
    add_param.output_offset = config.output_offset;

    if (graph->getGraphRootNode() == nullptr) {
      graph->setGraphRootNode(node); 
    }

    // Put this node's output tensor in map.
    graph->setTensor2Node(output_name, node);

    // Let predecessor node in topological order link to this node.
    auto pre_node = graph->getGraphTailNode();
    if(pre_node) {
    pre_node->next_ = node;
    }

    graph->setGraphTailNode(node);
    node->next_= nullptr;
    // Create node's device param.

    // Fill fpga_add_param.
    add_param.input1_c = i_dims_x[1];
    add_param.input1_h = i_dims_x[2];
    add_param.input1_w = i_dims_x[3];
    add_param.input2_c = i_dims_y[1];
    add_param.input2_h = i_dims_y[2];
    add_param.input2_w = i_dims_y[3];
    add_param.output_c = o_dims[1];
    add_param.output_h = o_dims[2];
    add_param.output_w = o_dims[3];
    add_param.input1_scale =  x_scales[0] / out_scales[0];
    add_param.input2_scale =  y_scales[0] / out_scales[0];
    add_param.output_scale = out_scales[0];
    add_param.type = (int)ElementWiseOpType::op_add;
    add_param.relu = ele_param->ac_type;

    CHECK(add_param.input1_offset >= 0);
    CHECK(add_param.input2_offset >= 0);
    CHECK(add_param.output_offset >= 0);
    VLOG(4) << "Converting elementwise op end.";
    
  return SUCCESS;
}

}  // namespace intel_fpga
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    fusion_elementwise_add_activation,
    kIntelFPGA,
    paddle::lite::subgraph::intel_fpga::ElementwiseConverter);
REGISTER_SUBGRAPH_BRIDGE(
    elementwise_add,
    kIntelFPGA,
    paddle::lite::subgraph::intel_fpga::ElementwiseConverter);
