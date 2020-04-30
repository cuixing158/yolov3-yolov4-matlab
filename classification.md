![RUNOOB 图标](https://github.com/cuixing158/yolov3-yolov4-matlab/blob/master/imagesResult/importerExporter.png)<br>
# Darknet Importer and Exporter
The importer can import all the seriesNetworks in the darknet and some simple DAGnetworks. The exporter can export all the seriesNetworks and some of the backbone networks. In addition to importing the deep neural network, the importer can obtain the feature map size of the network, the number of parameters, and the computational power FLOPs. For yolov2, yolov3 can also import a number of previous modules for later access to the yolo layer. 
This repositories requires Matlab2019a version and above, no other dependencies.

***

目前测试能导入和导出官网[darknet](https://github.com/pjreddie/darknet)以下网络，其他resnet*,yolo*可导入部分网络层<br>
alexnet.cfg<br>
cifar.cfg<br>
darknet.cfg <br>
darknet19.cfg<br>
darknet19_448.cfg<br>
darknet53.cfg <br>
darknet9000.cfg<br>
densenet201.cfg <br>
extraction.cfg<br>
extraction.conv.cfg<br>
extraction22k.cfg tiny.cfg<br>
tiny.cfg<br>
能够计算导入网络的特征图大小，FLOPs，参数个数。<br>
 
**1、示例一：导入模型示例**<br>
从https://pjreddie.com/darknet/imagenet/ 找到Pre_Trained Models的下载链接并点击下载，比如使用darknet19
  
```
cfg_file = 'darknet19.cfg';
weight_file = 'darknet19.weights';
[mynet,hyperParams,numsNetParams,FLOPs] = importDarknetNetwork(cfg_file,weight_file);
```
![RUNOOB 图标](https://github.com/cuixing158/yolov3-yolov4-matlab/blob/master/imagesResult/importDarknetNetwork.png)
```
analyzeNetwork(mynet)
```
![RUNOOB 图标](https://github.com/cuixing158/yolov3-yolov4-matlab/blob/master/imagesResult/mynet.png)
```
% 获取classesNames
fid = fopen('imagenet_shortnames_list.txt','r');
cls = textscan(fid, '%s', 'Delimiter',{'\n'});
classesNames = cls{1};
fclose(fid);

%  预测top5
img = imread('peppers.png');
input_size = mynet.Layers(1).InputSize;
img =im2single(imresize(img,input_size(1:2)));
scores = predict(mynet,img);
rank_k = 5;
[max_scores,ids] = maxk(scores,rank_k,2);
for i = 1:rank_k
    predictLabel = classesNames{ids(i)};
    predictScore = max_scores(i);
    fprintf('top%d, predictLabel:%-20s,predictScore:%.2f\n',i,predictLabel,predictScore);
end
```
![RUNOOB 图标](https://github.com/cuixing158/yolov3-yolov4-matlab/blob/master/imagesResult/rec_result.png)

**2、示例二：导入部分层和权重**<br>
```
cutoffModule = 15;
[lgraphWeight,hyperParams,numsNetParams,FLOPs] = importDarknetWeights(cfg_file,weight_file,cutoffModule);
analyzeNetwork(lgraphWeight)
```
![RUNOOB 图标](https://github.com/cuixing158/yolov3-yolov4-matlab/blob/master/imagesResult/importDarknetWeights.png)
![RUNOOB 图标](https://github.com/cuixing158/yolov3-yolov4-matlab/blob/master/imagesResult/lgraphWeight.png)
**3、示例三：导出mynet为cfg,weights 文件**<br>
```
hyperParams = [];
if isempty(hyperParams)
    hyperParams = struct('batch',64,...
        'subdivisiions',1,...
        'height',mynet.Layers(1).InputSize(1),...
        'width',mynet.Layers(1).InputSize(2),...
        'channels',mynet.Layers(1).InputSize(3),...
        'momentum',0.9,...
        'max_crop',256,...
        'learning_rate',0.01,...
        'policy','poly',...
        'power',4,...
        'max_batches',600000,...
        'angle',7,...
        'saturation',0.75,...
        'exposure',0.75,...
        'aspect',0.75);
end
% 指定导出的名字即可导出模型
cfgfile = 'exportC.cfg';
weightfile = 'exportW.weights';
exportDarkNetNetwork(mynet,hyperParams,cfgfile,weightfile)
```

**4、示例四：导入部分层示例**<br>
比如使用[yolov3-tiny.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-tiny.cfg)
```
cfgfile = 'yolov3-tiny.cfg';
cutoffModule = 17;
[lgraphLayer,hyperParams,numsNetParams,FLOPs,moduleTypeList,moduleInfoList,layerToModuleIndex] = importDarkNetLayers(cfgfile,cutoffModule);
analyzeNetwork(lgraphLayer)
```
![RUNOOB 图标](https://github.com/cuixing158/yolov3-yolov4-matlab/blob/master/imagesResult/importDarknetLayers.png)
![RUNOOB 图标](https://github.com/cuixing158/yolov3-yolov4-matlab/blob/master/imagesResult/lgraphLayer.png)

**5、示例五：导入导出第三方mobilenetV2模型**<br>
比如第三方的[mobilenetv2模型](https://github.com/cuixing158/pytorch-mobilenet-v2)来自于pytorch版本的。<br>
首先在pytorch模型转换onnx格式：<br>
```python
net = MobileNetV2(n_class = 1000)
state_dict = torch.load(r'mobilenet_v2.pth.tar')
net.load_state_dict(state_dict)
torch.onnx.export(net, torch.randn(1,3,224,224), "mobilenetv2_pytorch.onnx", verbose=True)
```
然后把“mobilenetv2_pytorch.onnx”导入到matlab中，如果遇到有的层不支持，则用[importONNXLayers](https://www.mathworks.com/help/deeplearning/ref/importonnxlayers.html)在deepNetworkDesigner中微调网络连接结构即可。<br>
```matlab
onnxf = 'mobilenetv2_pytorch.onnx';
lgraph = importONNXLayers(onnxf,'OutputLayerType','classification', ...
    'ImportWeights',true);
placeholderLayers = findPlaceholderLayers(lgraph);
revLayers = string(vertcat(placeholderLayers.Name));
lgraph = removeLayers(lgraph,revLayers)
deepNetworkDesigner()% 手动连接后导出
```
最终onnx导入到命名为m2.mat模型文件里,[百度网盘](https://pan.baidu.com/s/1i0NBPd9CzhyfC0m7t7Ri6w), 提取码: axnv ，用一图片验证其识别效果如下：<br>
另外再次导入导出（isLoadOnnx由true改为false）为cfg,weights的模型文件做验证，验证OK！
```matlab
classes = getClasses('synset_words.txt');
isLoadOnnx = true;
mobilenetv2_cfg = 'm2_export.cfg';
mobilenetv2_weight = 'm2_export.weights';
if isLoadOnnx
    load m2.mat
else %再导入已保存(matlab导出)的模型
    [netCNN,hyperParams,numsNetParams,FLOPs] = importDarknetNetwork(mobilenetv2_cfg,mobilenetv2_weight);
end
sz = netCNN.Layers(1).InputSize;

%% 图像预处理
image = imread('peppers.png');
image = im2single(imresize(image,sz(1:2)));
mean = [0.485, 0.456, 0.406];
std = [0.229, 0.224, 0.225];
image(:,:,1) = (image(:,:,1)-mean(1))/std(1);
image(:,:,2) = (image(:,:,2)-mean(2))/std(2);
image(:,:,3) = (image(:,:,3)-mean(3))/std(3);

%% predict
Ypredict = predict(netCNN,image);
[max_val,ind] = max(Ypredict);
predictLabel = classes(ind);
predictScore = max_val;
str = sprintf('predictLabel:%s, predictScore:%.5f\n',string(predictLabel),string(predictScore));
fprintf('%s',str); % 预测分数结果0.7757

%% 导出模型
mynet = netCNN;
hyperParams = [];
if isempty(hyperParams)
    hyperParams = struct('batch',64,...
        'subdivisiions',1,...
        'height',mynet.Layers(1).InputSize(1),...
        'width',mynet.Layers(1).InputSize(2),...
        'channels',mynet.Layers(1).InputSize(3),...
        'momentum',0.9,...
        'max_crop',256,...
        'learning_rate',0.01,...
        'policy','poly',...
        'power',4,...
        'max_batches',600000,...
        'angle',7,...
        'saturation',0.75,...
        'exposure',0.75,...
        'aspect',0.75);
end
exportDarkNetNetwork(mynet,hyperParams,mobilenetv2_cfg,mobilenetv2_weight)
```
预测结果：<br>
predictLabel:n07720875 bell pepper, predictScore:0.77573
