# How to Trust Unlabeled Data? Instance Credibility Inference for Few-Shot Learning

## 目录

- [1. 简介]()
- [2. 数据集和复现精度]()
- [3. 准备数据与环境]()
    - [3.1 准备环境]()
    - [3.2 准备数据]()
    - [3.3 准备模型]()
- [4. 开始使用]()
    - [4.1 模型训练]()
      - [4.1.1 fp32训练]()
      - [4.1.1 量化训练]()
    - [4.2 模型评估]()
    - [4.3 模型预测]()
    - [4.4 paddlelite转换]()
- [5. 模型推理部署]()
- [6. 自动化测试脚本]()
- [7. LICENSE]()
- [8. 参考链接与文献]()

## 1. 简介

​	   实例可信度推断模型聚焦如何将未标记样本更好加入到分类器的训练当中。其整体的模型见图，首先支持集中有标签标记样本集和未标记补充样本集经过一个预训练好的特征提取网络，得到各自对应的特征映射。其中含标签的样本将直接参与线性分类器的训练以得到一个初步的分类器模型，然后使用该分类器模型对未标记样本进行标签预测，得到对应的假标签 (pseudo label)。这些含假标签的样本集将通过实例可信度推断模块进行筛选，得到置信度高的假标签样本数据，并将这些数据添加到线性分类器模型的训练数据中再次训练分类器。以此不断往复训练、推断、添加，最终训练出一个拟合能力和泛化能力更强的线性分类器。最后，便可利用该训练好的分类器对询问集数据进行标签预测，完成小样本分类任务


![ICI](./images/img.png)


**论文:** [How to Trust Unlabeled Data? Instance Credibility Inference for Few-Shot Learning](https://arxiv.org/pdf/2007.08461.pdf)

**参考repo:** [ICI-FSL](https://github.com/Yikai-Wang/ICI-FSL/tree/master/V2-TPAMI)

在此非常感谢 ICI-FSL repo的Yikai-Wang等人贡献的[ICI-FSL](https://github.com/Yikai-Wang/ICI-FSL/tree/master/V2-TPAMI) ，提高了本repo复现论文的效率。



## 2. 数据集和复现精度
**数据集:** [下载](https://pan.baidu.com/s/1FPeqtzYBYHPu8ZhXpwKu_A?pwd=utv6)

miniImageNet数据集节选自ImageNet数据集,包含100类共60000张彩色图片，其中每类有600个样本，每张图片的规格为84 × 84 。通常而言,这个数据集的训练集和测试集的类别划分为：80 : 20。
**复现精度:**
本次轻量化的实现方式为使用[paddleslim](https://github.com/PaddlePaddle/PaddleSlim) 量化训练

| ICIR | 1shot  | 5shot  | 模型尺寸 |
|------|--------|--------|------|
| 论文   | 72.25% | 83.25% | -    |
| 复现   | 72.44% | 83.38% | 31M  |
| 量化   | 72.45% | 83.43% | 7.7M |

### 模型下载
#### 下载地址
[百度网盘](https://pan.baidu.com/s/1FPeqtzYBYHPu8ZhXpwKu_A?pwd=utv6)
#### 文件说明
- ckpt.zip：包含动态图训练的模型，方便复现量化训练  
- 1shot.zip和5shot.zip包含以下文件：  
├── best_model.tar # 动态图训练的模型  
├── inference.pdiparams # 量化后导出的静态模型  
├── inference.pdmodel # 量化后导出的静态模型  
├── lite.nb # paddlelite导出的实际大小的静态模型  


## 3. 准备数据与环境


### 3.1 准备环境

首先介绍下支持的硬件和框架版本等环境的要求，格式如下：

- 硬件：GPU: Tesla V100 Mem 16GB, CPU 2cores RAM 16GB (aistudio高级GPU)
- 框架：
  - paddlepaddle-gpu===2.3.0.post101
- 使用如下命令安装依赖：

```bash
python -m pip install paddlepaddle-gpu==2.3.0.post101 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.htmlpip install paddleslim
pip install paddleslim
pip install paddlelite
pip install glmnet-py
## 安装AutoLog（规范化日志输出工具）
pip install  https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
```

### 3.2 准备数据

数据解压到data目录下，少量测试数据sample_data已包含在本repo

```
# 全量数据： data/MiniImagenet
# 少量数据： data/sample_data
```

## 4. 开始使用

### 4.1 模型训练
#### 4.1.1 fp32训练
```bash
# 训练 1shot
python main.py --dataset miniImageNet --save-dir ckpt/miniImageNet/1-shot -g 0 --nKnovel 5 --nExemplars 1 --phase val --mode train
# 训练 5shot
python main.py --dataset miniImageNet --save-dir ckpt/miniImageNet/5-shot -g 0 --nKnovel 5 --nExemplars 5 --phase val --mode train
```

#### 4.1.2 量化训练
fp32模型没有重新训练，沿用了之前的模型。开始前把**ckpt.zip解压到当前目录下**
```bash
# 量化训练 1shot 
python main_quant.py --dataset miniImageNet --save-dir ckpt/miniImageNet/1-shot -g 0 --nKnovel 5 --nExemplars 1 --phase val --mode train --resume ckpt/miniImageNet/1-shot/best_model.tar --max-epoch 1 --model_dir 1shot_quat/inference
# 量化训练 5shot
python main_quant.py --dataset miniImageNet --save-dir ckpt/miniImageNet/5-shot -g 0 --nKnovel 5 --nExemplars 5 --phase val --mode train --resume ckpt/miniImageNet/5-shot/best_model.tar --max-epoch 1 --model_dir 5shot_quat/inference
```

### 4.2 模型评估
评估量化后的模型
```bash
# 评估 1shot
python test_by_infer.py --dataset_dir ./data/MiniImagenet --benchmark False --model_dir 1shot_quat --nKnovel 5 --nExemplars 1 --phase test --mode test
# 评估 5shot 
python test_by_infer.py --dataset_dir ./data/MiniImagenet --benchmark False --model_dir 5shot_quat --nKnovel 5 --nExemplars 5 --phase test --mode test
```
评估结果
```
100% 2000/2000 [32:27<00:00,  1.03it/s] 
80.97 82.61 83.29 83.44 83.43
0.316 0.319 0.327 0.331 0.331
```

### 4.3 模型预测
提取单张图片向量
<img src="./images/cat.jpg" width="10%" height="1%" />
```bash
python main.py --dataset miniImageNet --save-dir ckpt/miniImageNet/test  --mode predict --resume ckpt/miniImageNet/1-shot/best_model.tar
# img embedding extracted, shape is (1, 512)
```
### 4.4 paddlelite转换
把int8量化模型转成lite格式，查看模型实际大小
```bash
paddle_lite_opt \
    --model_file="1shot_quat/inference.pdmodel"  \
    --param_file="1shot_quat/inference.pdiparams" \
    --optimize_out=./1shot/lite \
    --quant_model=true \
    --quant_type=QUANT_INT8
```

## 5. 模型推理部署
###  模型导出
```bash
python3 export_model.py \
--resume ckpt/miniImageNet/1-shot/best_model.tar \
--output_path ckpt/miniImageNet/1-shot/
```
### 静态图推理
提取单张图片向量，输入同动态图
```bash
python3 infer.py --model_dir 1shot_quat/
# img embedding extracted, shape is (1, 512)
```


## 6. 自动化测试脚本

```shell
bash test_tipc/test_train_inference_python.sh test_tipc/configs/train_infer_python.txt  lite_train_lite_infer
```
关键步骤展示
![](./images/tipc1.png)
![](./images/tipc2.png)
## 7. LICENSE

本项目的发布受[Apache 2.0 license](./LICENSE)许可认证。

## 8. 参考链接与文献

[1] [How to Trust Unlabeled Data? Instance Credibility Inference for Few-Shot Learning](https://arxiv.org/pdf/2007.08461.pdf)

[2] [ICI-FSL](https://github.com/Yikai-Wang/ICI-FSL/tree/master/V2-TPAMI)


