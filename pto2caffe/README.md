## PTO2Caffe

Convert a deep learning model from Pytorch,TensorFlow, TFLite, ONNX to Caffe.

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
python convert.py -platform=pytorch -model=/path/to/torchscript/model.pt -root_folder=/path/to/input_data/ -mean 0.485 0.456 0.406 -scale 255 255 255 -std 0.229 0.224 0.225 -bin_shape 3 2560 1440 -compare=1 -crop_h=480 -crop_w=480
```


**Required parameter:**

- **platform:** Specify the original model type, include Pytorch, Tensorflow, TFLite, ONNX

- **model:** The path of original model file.

- **root_folder:** Specify the input data root folder.


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



#### **Operators:**

**Tensorflow:**

| Operators             | Comment |
| --------------------- | ------- |
| Pad                   |         |
| Add                   |         |
| Mul                   |         |
| AddV2                 |         |
| MaxPool               |         |
| BiasAdd               |         |
| ConcatV2              |         |
| Placeholder           |         |
| Conv2D                |         |
| LeakyRelu             |         |
| FusedBatchNormV3      |         |
| ResizeNearestNeighbor |         |
| SpaceToDepth          |         |



**TFLite:**

| Operators               |      |
| ----------------------- | ---- |
| PAD                     |      |
| ADD                     |      |
| MUL                     |      |
| SUB                     |      |
| SPLIT                   |      |
| MEAN                    |      |
| RESHAPE                 |      |
| SQUEEZE                 |      |
| SOFTMAX                 |      |
| RELU                    |      |
| PRELU                   |      |
| HARD_SWISH              |      |
| QUANTIZE                |      |
| TRANSPOSE               |      |
| CONV_2D                 |      |
| LOGISTIC                |      |
| DEQUANTIZE              |      |
| MAX_POOL_2D             |      |
| CONCATENATION           |      |
| REDUCE_MAX              |      |
| LEAKY_RELU              |      |
| RESIZE_BILINEAR         |      |
| AVERAGE_POOL_2D         |      |
| TRANSPOSE_CONV          |      |
| FULLY_CONNECTED         |      |
| DEPTHWISE_CONV_2D       |      |
| RESIZE_NEAREST_NEIGHBOR |      |



**ONNX:**

| Operators             |      |
| --------------------- | ---- |
| Exp                   |      |
| Log                   |      |
| Pad                   |      |
| LRN                   |      |
| Add                   |      |
| Sub                   |      |
| Mul                   |      |
| Pow                   |      |
| Tanh                  |      |
| Sqrt                  |      |
| Sum                   |      |
| Div                   |      |
| Slice                 |      |
| Split                 |      |
| MatMul                | Only support input dimentions == 2 |
| Concat                |      |
| Resize                |      |
| Dropout               |      |
| Reshape               |      |
| Squeeze               |      |
| Flatten               |      |
| MaxPool               |      |
| Relu                  |      |
| Clip                  |      |
| Conv                  |      |
| PRelu                 |      |
| Identity              |      |
| Gemm                  |      |
| Constant              |      |
| Softplus              |      |
| Unsqueeze             |      |
| Transpose             |      |
| ReduceMean            |      |
| Sigmoid               |      |
| Softmax               |      |
| AveragePool           |      |
| LeakyRelu             |      |
| GlobalAveragePool     |      |
| ConvTranspose         |      |
| BatchNormalization    |      |
| InstanceNormalization |      |
| Upsample              |      |
| Mish                  |      |



**Pytorch:**

| Operators    |      |
| ------------ | ---- |
| nn.ReLU      |      |
| torch.cat    |      |
| nn.Linear    |      |
| nn.Dropout   |      |
| Tensor.slice |      |
| nn.AvgPool2d |      |
| nn.MaxPool2d |      |
| nn.Conv2d    |      |

