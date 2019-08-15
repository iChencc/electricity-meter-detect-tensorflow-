# 电表类型探测（tensorflow小实战）


## 1. 简介
实现对于高压作业中实现机器视野自动识别电表或者一些其他仪器的算法设计，就是一个图像处理，随着深度学习近年来的发展，对于图像处理算法的设计深度学习是一个很好的选择。这个实验就是基于tensorflow框架实现的

## 2. 目录结构

```c
|---test                /*测试的python代码*/
|   |---result_photo    /*储存测试结果图片*/    
|   |---save            /*存放tensorflow的Saver*/
|   |---test_photo      /*测试用的图片*/
|---train               /*训练的python代码*/
|   |---save            /*存放tensorflow的Saver*/
|   |---train_data      /*训练用的图片*/
```

## 3. 需要环境
我是在
