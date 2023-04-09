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

#include "lite/operators/pool_op.h"
#include<intelfpga.h>
#include "lite/kernels/intel_fpga/bridges/graph.h"
#include "lite/core/subgraph/subgraph_bridge_registry.h"

namespace paddle {
namespace lite {
namespace subgraph {
namespace intel_fpga {

int PoolConverter(void *ctx, OpLite *op, KernelBase *kernel) {
    CHECK(ctx != nullptr);
    CHECK(op != nullptr);
    auto graph = static_cast<Graph*>(ctx);

    VLOG(4) << "Converting pool2d op for intelfpga.";
    operators::PoolParam& param = kernel->Param<operators::PoolParam>();
    std::string op_type = op->op_info()->Type();
    auto op_info = op->op_info();
    auto scope = op->scope();

    auto input_name = op_info->Input("X").front();
    auto output_name = op_info->Output("Out").front();
    auto i_dims = param.x->dims();
    auto o_dims = param.output->dims();

    Node* node = new Node();
    node->name_ = output_name;
    FpgaConvParam* device_param = new FpgaConvParam();
    node->node_param_ = dynamic_cast<NodeParam*>(device_param);
    node->is_input = graph->IsInput(input_name);
    node->is_output = graph->IsOutput(output_name);
    node->op_type_ = INTELFPGA_Conv2D;

    // Find this node's parent according to input tensor.
    if(graph->GetNodeByTensorName(input_name)) {
        node->parent_vec_.push_back(graph->GetNodeByTensorName(input_name));
        int byte_offset =
          FpgaWord2ByteOffset(node->parent_vec_[0]->op_type_,
              FpgaGetOutputOffset(node->parent_vec_[0]));
      device_param->param.input_offset = FpgaByte2WordOffset(node->op_type_, byte_offset);
    } else {
      node->parent_vec_.push_back(nullptr);
      device_output_config config = FpgaMemMalloc(node->op_type_,
          device_param->d_ia, i_dims[1], i_dims[2], i_dims[3]);
      device_param->param.input_offset = config.output_offset;
    }
    if (graph->getGraphRootNode() == nullptr) {
      graph->setGraphRootNode(node);
    } 

    // Malloc output and set offset.
    device_output_config config = FpgaMemMalloc(node->op_type_,
        device_param->d_oa, o_dims[1], o_dims[2], o_dims[3]);
    device_param->param.output_offset = config.output_offset;
    device_param->param.output_size = config.output_size;
    VLOG(4) << "output_offset: " << device_param->param.output_offset;
    VLOG(4) << "output_siez: " << device_param->param.output_size;
    auto x_scale_name = "X0_scale";
    std::vector<float> x_scales;
    if (op_info->HasInputScale(x_scale_name, true)) {
      x_scales = op_info->GetInputScale(x_scale_name, true);
    }
    VLOG(4) << "X0_scale: " << x_scales[0];
    auto out_scale_name = "Out0_scale";
    std::vector<float> out_scales;
     if (op_info->HasOutputScale(out_scale_name, true)) {
      out_scales = op_info->GetOutputScale(out_scale_name, true);
    }
    VLOG(4) << "Out0_scale: " << out_scales[0];
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

    device_param->ia = (int8_t*)(param.x->data<int8_t>());
    device_param->oa = (int8_t*)(param.output->mutable_data<float>());
    
    // Fill fpga_param.
    auto paddings = *param.paddings; 
    //init scale
    #if (FPGA_SOURCE == FPGA_SOURCE_NK)
    device_param->scale = new float[2+2*o_dims[1]];
    device_param->scale[0]= x_scales[0];
    device_param->scale[1]= out_scales[0];
    #elif (FPGA_SOURCE == FPGA_SOURCE_AW)
    device_param->scale = new float[2*o_dims[1]];
    device_param->param.input_scale = x_scales[0];
    device_param->param.output_scale = out_scales[0];
    #endif
    
    for(int i=0;i<o_dims[1];i++)
    {
      #if (FPGA_SOURCE == FPGA_SOURCE_NK)
      device_param->scale[2+o_dims[1]+i]=0;
      #elif (FPGA_SOURCE == FPGA_SOURCE_AW)
      device_param->scale[o_dims[1]+i]=0;
      #endif
    }
    #if (FPGA_SOURCE == FPGA_SOURCE_NK)
    device_param->param.scale_offset = 2;
    #elif (FPGA_SOURCE == FPGA_SOURCE_AW)
    device_param->param.scale_offset = 0;
    #endif
    // device_param->param.weight_offset = 0;
    device_param->param.in_c=i_dims[1];
    device_param->param.in_h=i_dims[2];
    device_param->param.in_w=i_dims[3];
    device_param->param.output_c=o_dims[1];
    device_param->param.output_h=o_dims[2];
    device_param->param.output_w=o_dims[3];
    device_param->param.in_pad=paddings[0];
    device_param->param.stride=param.strides[0];
    device_param->param.kernel=param.ksize[0];
    // device_param->ip.dy = dilations[0];
    // device_param->ip.dx = dilations[1];
    if(param.pooling_type == "max"){
        device_param->param.type=INTELFPGA_Pool2D_MAX;
    }
    else{
        device_param->param.type=INTELFPGA_Pool2D_AVG;
    }
    
    VLOG(4) << "Converting pool2d op for intelfpga end.";
  return SUCCESS;
}

}  // namespace intel_fpga
}  // namespace subgraph
}  // namespace lite
}  // namespace paddle

REGISTER_SUBGRAPH_BRIDGE(
    pool2d,
    kIntelFPGA,
    paddle::lite::subgraph::intel_fpga::PoolConverter);

