# 机器学习第一次作业：图像分类作业   
<h3 style="text-align: center;">
  张韶恒 2023312309 自动化三班
</h3>

>  ---
> ## 作业题目内容
> ### 环境
>
>   - [x] Pytorch
>   - [x] food11数据
>
> > 尽量在Ubuntu下运行
> > 程序在Windows下运行, 也可以在Ubuntu下，注意修改代码中的路径分隔符。
> > 数据在 [https://www.kaggle.com/competitions/ml2022spring-hw3b/data](https://www.kaggle.com/competitions/ml2022spring-hw3b/data) 上下载
>
> ### 代码
>
> ```py
> # Normally, We don't need augmentations in testing and validation.
> # All we need here is to resize the PIL image and transform it into Tensor.
> test_tfm = transforms.Compose([
>     transforms.Resize((128, 128)),
>     transforms.ToTensor(),
> ])
> # However, it is also possible to use augmentation in the testing phase.
> # You may use train_tfm to produce a variety of images and then test using ensemble methods
> train_tfm = transforms.Compose([
>     # Resize the image into a fixed shape (height = width = 128)
>     transforms.Resize((128, 128)),
>     # You may add some transforms here.
>     # ToTensor() should be the last one of the transforms.
>     transforms.ToTensor(),
> ])
> class FoodDataset(Dataset):
>     ...
> ```
>
> 处理图片数据的代码块
>
> ```py
> class Classifier(nn.Module):
>     ...
> ```
>
> 分类器定义，可以设置网络大小和结构
>
> ```py
> def Training_Demo():
>     ...
> ```
>
> 训练函数，定义各种参数并且训练模型。
>
> ```py
> def Testing_Demo():
>     ...
> ```
>
> 测试函数，测试模型的准确率
>
> ```py
> def Predict_Demo():
>     ...
> ```
>
> 使用模型进行预测分类
>
> ### 效果
>
> | | 准确率 |
> | --- | --- |
> | 训练集 | 0.63411 |
> | 测试集 | 0.56291 |
>---
---
## 解答思路（2 stage）

我将此project中的三个文件分别称为：
* **`oringin.py`**: 原始版本(xh老师给的)
* **`improved.py`**: 数据增强版（仅改变参数&增强数据）
* **`final_version.py`**: 迁移学习最终版 (迁移了 `EfficientNet-B0`模型的 版本)

---

### 1. `improved.py` 较 `oringin.py` 的优化

这个阶段的优化，核心在于**实现代码平台兼容、增加数据、改进基础训练策略**。

* **【实现代码平台兼容】跨平台兼容性**：
    * `oringin.py` 的代码硬编码了Linux的路径分隔符 `/`，导致在Windows上无法正确提取标签。
    * `improved.py` 引入了 `os.path.basename()`，使得标签提取逻辑能够同时兼容Windows和Linux，这是一个**从“只兼容Linux”到“Win和Linux均可以运行”的优化**。

* **【核心性能优化】引入数据增强 (Data Augmentation)**：
    * `oringin.py` 的训练数据处理 (`train_tfm`) 非常简单，仅仅是缩放尺寸，模型每一轮看到的图片都是一模一样的。
    * `improved.py` 在 `train_tfm` 中加入了**随机翻转、随机旋转、随机色彩变换**。这使得模型在训练时，能看到远比原始数据集更丰富、更多样的图片，是**提升模型泛化能力、防止过拟合的关键第一步**。

* **【训练策略优化】更合理的训练参数**：
    * `oringin.py` 只训练了5轮 (`n_epochs = 5`)，并且`patience`值设得过高(300)，导致训练不充分且无法有效提前停止。
    * `improved.py` 将训练轮数增加到了更合理的**30轮**，并设置了`patience = 5`，让模型有充足的时间学习，同时能在遇到瓶颈时及时停止，节省时间。
    * 优化器从 `Adam` 升级到了 `AdamW`，后者在处理权重衰减时通常表现更好。

* **【模型结构调整】**：
    * `oringin.py` 使用了一个更深（5个池化层）、通道数更多（达到512）的自定义CNN。
    * `improved.py` 使用了一个相对更浅（3个池化层）、更“标准”的自定义CNN结构。**（适应n_epoches增大带来的显存负担）**

---

### 2. `final_version.py` (迁移学习版) 较 `improved.py` 的优化

这个阶段是一次***质的飞跃***，我可以说是从“自己摸索”的层面，跃升到了“站在巨人肩膀上”的业界标准方法（maybe）。

* **【核心架构革命】引入迁移学习 (Transfer Learning)**：
    * `improved.py` 仍然是依赖一个**从零开始训练**的自定义CNN，其能力上限受限于模型结构和数据量。
    * `final_version.py` 彻底放弃了自定义模型，直接采用了在ImageNet（百万级数据集）上**预训练好的 `EfficientNet-B0` 模型**。这使得我们的模型天生就具备了强大的图像特征提取能力，我们只需要微调其最后一层来适应我们的食物分类任务即可。**这是冲击高准确率的最关键改动**。

* **【数据处理升级】适配预训练模型**：
    * `improved.py` 的输入尺寸是`128x128`。
    * `final_version.py` 将输入尺寸改为了`224x224`，并加入了`Normalize`（标准化）步骤。这完全是为了**匹配 `EfficientNet-B0` 的“胃口”**，因为这些预训练模型就是用这种尺寸和标准化的数据进行训练的，这样做能最大化地发挥其性能。

* **【训练策略进阶】更先进的学习率调度**：
    * `improved.py` 没有使用学习率调度器。
    * `final_version.py` 引入了**余弦退火学习率调度器 (`CosineAnnealingLR`)**。它能让学习率在训练过程中以一种平滑的、类似余弦曲线的方式进行衰减，这通常比固定学习率或者阶梯式下降的学习率效果更好，有助于模型找到更优的解。

* **【正则化增强】**：
    * 在`EfficientNet`的分类器头部加入了`Dropout`层，在训练时随机丢弃一些神经元，这是另一种非常有效的**防止过拟合**的手段。

---

### 总结：`final_version.py` 较 `oringin.py` 的完整优化链条

`final_version.py` 相比于最初的 `oringin.py`，完成了一整套从**基础修正**到**高级优化**的全面升级：

1.  **优化实现了跨平台的代码运行**，让代码能在Windows上正确运行。
2.  **引入了丰富的数据增强**，大大提升了数据集的有效规模和多样性。
3.  **采用了更合理的训练周期和提前停止策略**，让训练过程更高效。
4.  **最终，用更优秀的“迁移学习”方案（`EfficientNet-B0`）替换了简单的自定义模型**，这是实现性能突破的决定性一步。
5.  **相应地，将数据预处理和训练策略（如学习率调度）也升级到了与迁移学习相匹配的更高水准**。
---
### 三者效果展示
| | oringin.py的准确率 | improved.py的准确率 |final_version.py的准确率 |
| --- | --- | --- | --- |
| 训练集 | 0.634 |0.734|0.748|
| 测试集 | 0.563 |0.640|0.810|
>`final_version.py`的training中使用的数据集为train_tfm，做了较为复杂的处理，为识别增加了困难，因此训练集准确率较低，然而测试集采用的是test_tfm，数据集仅做了中心裁剪和尺寸缩放，处理简单，因此准确率较高。（类似于考前模拟训练考的难分低，真高考了反而题目简单分高点）   

>我们可以发现，经过上述优化，最终版的模型在训练集和测试集上的准确率都有显著提升，最终达到了0.932和0.808，这表明通过引入迁移学习、数据增强和更先进的训练策略，我们可以显著提高模型性能。
 最终获得的测试集准确度大于老师课上要求的0.7，成功！

 #### 运行结果
 * `improved.py`   运行结果：
![alt text](image.png)
* `final_version.py`  运行结果：
  ![alt text](image-1.png)