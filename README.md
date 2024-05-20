# ChatGLM3-Finetuning
A chinese finetuning example code for ChatGLM3 model, offering traing and predict code and basic usage.

# Lora
- 在原始 PLM (Pre-trained Language Model) 旁边增加一个旁路，做一个降维再升维的操作，来模拟所谓的intrinsic rank；
- 训练的时候固定 PLM 的参数，只训练降维矩阵 AAA 与升维矩阵 BBB 。而模型的输入输出维度不变，输出时将 BABABA 与 PLM 的参数叠加；
- 用随机高斯分布初始化 AAA ，用 0 矩阵初始化 BBB ，保证训练的开始此旁路矩阵依然是 0 矩阵。

# DeepSpeed
DeepSpeed是微软发布的深度学习训练框架，它旨在解决大模型数据并行训练时显存溢出以及模型难以并行的问题，通过在计算、通信、显存内存、IO以及超参的组合优化来提升训练性能。
```
pip install deepspeed
```

# Train
可以直接通过执行shell文件来进行训练，参数可以对shell直接进行修改
```
train.sh
```

# Predict
传入lora模型的地址，和pretrained模型的地址，代码会自动将模型进行合并然后进行predict
```
python predict.py
```
