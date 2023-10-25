## Pto2Caffe

Convert a deep learning model from Pytorch,TensorFlow, TFLite, ONNX, Tvm to Caffe.

#### 1. Dependency

To use pto2caffe, you need

1. Docker image onlyxool/vimicro-ai
[onlyxool/vimicro-ai:gpu](https://hub.docker.com/r/onlyxool/vimicro-ai)
[onlyxool/vimicro-ai:cpu](https://hub.docker.com/r/onlyxool/vimicro-ai)

   ```bash
   docker pull onlyxool/vimicro-ai:gpu
   docker pull onlyxool/vimicro-ai:cpu
   ```

3. Download [Caffe](https://10.0.13.134/ai_npu/caffe) source code then build it.



#### 2. Arguments

​	A common usage example as below:

```bash
python convert.py -platform=onnx -model=/path/to/onnx/model.onnx -compare=1
```

**Required parameter:**

- **platform:** Specify the original model type, include Pytorch, Tensorflow, TFLite, ONNX

- **model:** The path of original model file.


**Input Raw Data:**

- **bin_shape:** specify the input shape. default layout is [C,H,W]

- **dtype:**  specify the Data type -dtype=0: u8, -dtype=1: s16, -dtype2: f32 


**Input Image:**

- **color_format:** Specify the images color format, 0: BGR 1: RGB 2: GRAY. Default: RGB

**Data pre-process:**

- **mean:** Then number of mean value has to be the same number of channels
- **scale:** Then number of scale value has to be the same number of channels
- **std:** Then number of std value has to be the same number of channels
- **crop_h & crop_w:** Specify if you would like to centrally crop input image

**Option Arguments:**

- **simplifier:** simplify onnx model by onnx-simplifier
- **auto_crop:** Crop the input data according to the model inputs size. Can't apply pytorch model.
- **log:** log print level, 0: Debug 1: Info 2: Warning, 3: ERROR
- **compare:** Compare network output, 0: Compare latest layer 1: Compare every layer
- **root_folder:** Specify the input data root folder.


#### **Operators:**

**Tensorflow:**

| Operators             | Comment |
| --------------------- | ------- |
| Pad                   |         |
| Add                   |         |
| Sub                   |         |
| Mul                   |         |
| AddV2                 |         |
| Relu6                 |         |
| MatMul                |         |
| MaxPool               |         |
| AvgPool               |         |
| BiasAdd               |         |
| Squeeze               |         |
| Reshape               |         |
| Softmax               |         |
| ConcatV2              |         |
| Placeholder           |         |
| Conv2D                |         |
| LeakyRelu             |         |
| FusedBatchNormV3      |         |
| SpaceToDepth          |         |
| ResizeNearestNeighbor |         |
| DepthwiseConv2dNative |         |



**TFLite:**

| Operators               |      |
| ----------------------- | ---- |
| PAD                     |      |
| ADD                     |      |
| MUL                     |      |
| SUB                     |      |
| MEAN                    |      |
| RELU                    |      |
| PRELU                   |      |
| RELU6                   |      |
| SPLIT                   |      |
| CONV_2D                 |      |
| RESHAPE                 |      |
| SQUEEZE                 |      |
| SOFTMAX                 |      |
| LOGISTIC                |      |
| QUANTIZE                |      |
| TRANSPOSE               |      |
| DEQUANTIZE              |      |
| REDUCE_MAX              |      |
| HARD_SWISH              |      |
| LEAKY_RELU              |      |
| MAX_POOL_2D             |      |
| CONCATENATION           |      |
| TRANSPOSE_CONV          |      |
| DEPTH_TO_SPACE          |      |
| RESIZE_BILINEAR         |      |
| AVERAGE_POOL_2D         |      |
| FULLY_CONNECTED         |      |
| RESIZE_BILINEAR         |      |
| DEPTHWISE_CONV_2D       |      |
| RESIZE_NEAREST_NEIGHBOR |      |



**ONNX:**

| Operators             |      |
| --------------------- | ---- |
| Elu                   |      |
| Exp                   |      |
| Log                   |      |
| Pad                   |      |
| LRN                   |      |
| Add                   |      |
| Sum                   |      |
| Sub                   |      |
| Mul                   |      |
| Div                   |      |
| Pow                   |      |
| Tanh                  |      |
| Sqrt                  |      |
| Relu                  |      |
| Clip                  |      |
| Conv                  |      |
| Gemm                  |      |
| Mish                  |      |
| PRelu                 |      |
| Slice                 |      |
| Split                 |      |
| MatMul                |      |
| Concat                |      |
| Resize                |      |
| Dropout               |      |
| Reshape               |      |
| Squeeze               |      |
| Flatten               |      |
| MaxPool               |      |
| Sigmoid               |      |
| Softmax               |      |
| Identity              |      |
| Constant              |      |
| Softplus              |      |
| Upsample              |      |
| LeakyRelu             |      |
| Unsqueeze             |      |
| Transpose             |      |
| ReduceMean            |      |
| AveragePool           |      |
| ConvTranspose         |      |
| GlobalAveragePool     |      |
| BatchNormalization    |      |
| InstanceNormalization |      |



**Pytorch:**

| Operators         |
| ----------------- |
| Pad               |
| Mul               |
| Add               |
| View              |
| Silu              |
| ReLU              |
| Mean              |
| Slice             |
| Conv2d            |
| Concat            |
| Linear            |
| Select            |
| MatMul            |
| Softmax           |
| Sigmoid           |
| Dropout           |
| Pooling           |
| Flatten           |
| Permute           |
| Reshape           |
| Upsample          |
| Transpose         |
| Unsqueeze         |
| Hardswish         |
| BatchNorm2d       |
| ConvTranspose2d   |
| AdaptiveAvgPool2d |
