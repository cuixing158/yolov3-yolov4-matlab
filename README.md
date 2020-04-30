![RUNOOB 图标](https://github.com/cuixing158/yolov3-yolov4-matlab/blob/master/images/importerExporter.png)<br>
![RUNOOB 图标](https://github.com/cuixing158/yolov3-yolov4-matlab/blob/master/images/yolov4Detect.jpg)<br>
![RUNOOB 图标](https://github.com/cuixing158/yolov3-yolov4-matlab/blob/master/images/dogYolov4Detect.jpg)<br>

# yoloV3/yolov4 matlab
This respository uses simplified and minimal code to reproduce the yolov3 / yolov4 detection networks and darknet classification networks. The highlights are as follows:<br>
- 1、Support original version of darknet model；<br>
- 2、Support training, inference, import and export of "* .cfg", "* .weights" models；<br>
- 3、Support the latest [yolov3, yolov4](https://github.com/AlexeyAB/darknet) models;<br>
- 4、Support darknet classification model;**<br>
- 5、Support all kinds of indicators such as feature map size calculation, flops calculation and so on.<br>
- These code is highly readable and more brief than other frameworks such as pytorch and tensorflow!<br>
但是不包含各种训练数据增强tricks！


# Requirements
Matlab R2019b or higher,the newer the better,no other dependencies!!!

# How to use
训练使用train.m,推理测试使用detect.m,模型分类参考mainClassification.mlx<br>
百度网盘yolov3/4,[weights文件](https://pan.baidu.com/s/1UvPKSlT7K3hzeXof4ovN_A) 提取码：dbbo
