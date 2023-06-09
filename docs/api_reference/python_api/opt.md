## Opt

```python
class Opt;
```

`Opt`模型离线优化接口，Paddle原生模型需经`opt`优化图结构后才能在Paddle Lite上运行。

示例：  

假设待转化模型问当前文件夹下的`mobilenet_v1`，可以使用以下脚本转换

```python
# 引用Paddlelite预测库
from paddlelite.lite import *

# 1. 创建opt实例
opt=Opt()
# 2. 指定输入模型地址 
opt.set_model_dir("./mobilenet_v1")
# 3. 指定转化类型： arm、x86、opencl、npu
opt.set_valid_places("arm")
# 4. 指定模型转化类型： naive_buffer、protobuf
opt.set_model_type("naive_buffer")
# 4. 输出模型地址
opt.set_optimize_out("mobilenetv1_opt")
# 5. 执行模型优化
opt.run()
```

### `set_model_dir(model_dir)`

设置模型文件夹路径，当需要从磁盘加载非combined模型时使用。

参数：

- `model_dir(str)` - 模型文件夹路径

返回：`None`



### `set_model_file(model_file)`

设置模型文件路径，加载combined形式模型时使用。

参数：

- `model_file(str)` - 模型文件路径



### `set_param_file(param_file)`

设置模型参数文件路径，加载combined形式模型时使用。

参数：

- `param_file(str)` - 模型文件路径


### `set_model_type(type)`

设置模型的输出类型，当前支持`naive_buffer`和`protobuf`两种格式，移动端预测需要转化为`naive_buffer`。

`naive_buffer`的优化后模型为以`.nb`名称结尾的单个文件。`protobuf`的优化后模型为文件夹下的`model`和`params`两个文件。将`model`重命名为`__model__`用[Netron](https://lutzroeder.github.io/netron/)打开，即可查看优化后的模型结构。

参数：

- `type(str)` - 模型格式（`naive_buffer/protobuf`)



### `set_valid_places(valid_places)`

设置可用的places列表。

参数：

- `valid_places(str)` - 可用place列表，不同place用`,`隔开

示例：

```python
# 引用Paddlelite预测库
from paddlelite.lite import *

# 1. 创建opt实例
opt=Opt()
# 2. 指定转化类型： arm、x86、opencl、npu
opt.set_valid_places("arm,opencl")
# opt.set_valid_places("arm,npu")
```




### `set_optimize_out(optimized_model_name)`

设置优化后模型的名称，优化后模型文件以`.nb`作为文件后缀。

参数：

- `optimized_model_name(str)`


### `set_quant_model(quant_model)`

设置是否使用 `opt` 中的动态离线量化功能。

参数：

- `quant_model(bool)`

### `set_quant_type(quant_type)`

设置动态离线量化的方式，支持两种量化方式（`QUANT_INT16`和`QUANT_INT8`），即分别量化为`int16`和`int8`。量化为`int8`对模型精度有一点影响，模型体积大概减小4倍。量化为`int16`对模型精度基本没有影响，模型体积大概减小2倍。

参数：

- `quant_type(str)`-支持设置为`QUANT_INT16`和`QUANT_INT8`

### `enable_fp16()`

启用`Float16 训练后量化`。将模型中权重数据量化为`Float16`对模型精度有一点影响，运行耗时和内存占用几乎降低一半。


### `set_sparse_model(bool)`

设置是否使用 `opt` 中的模型稀疏化功能。此功能目前只可以在ARM平台编译模型时开启。

参数：

- `sparse_model(bool)`

### `set_sparse_threshold(float)`

设置稀疏化阈值。例如，设置为0.6时，意思是对于某层参数，如果该参数的0元素比例小于0.6的话，则不走`sparse pass`。

参数：

- `sparse_threshold(float)`-支持参数区间为`[0,1]`


### `run()`

执行模型优化，用以上接口设置完 `模型路径`、`model_type`、`optimize_out`和`valid_places`后，执行`run()`接口会根据以上设置转化模型，转化后模型保存在当前路径下。


### `run_optimize(model_dir, model_file, param_file, type, valid_places, optimized_model_name)`

执行模型优化，无需设置以上接口，直接指定 `模型路径`、`model_type`、`optimize_out`和`valid_places`并执行模型转化。

参数：

- `model_dir(str)` - 模型文件夹路径
- `model_file(str)` - 模型文件路径
- `param_file(str)` - 模型文件路径
- `type(str)` - 模型格式（`naive_buffer/protobuf`)
- `valid_places(str)` - 可用place列表，不同place用`,`隔开
- `optimized_model_name(str)`

```python
# 引用Paddlelite预测库
from paddlelite.lite import *
# 1. 创建opt实例
opt=Opt()
# 2. 执行模型优化
opt.run_optimize("./mobilenet_v1","","","arm","mobilenetv1_opt");
```
