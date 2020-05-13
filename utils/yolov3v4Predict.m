function outPutFeatures = yolov3v4Predict(cfg_file,weight_file,image)
% 功能：yolov3快速输出检测特征，同darknet官网输出结果方式保持一致
% 输入：
%      cfg_file, 指定的cfg后缀的模型描述文件
%      weight_file, 指定的.weights后缀的二进制文件
%      image ：输入网络的图像数据,单张图像(H*W*C)或者批量图像(H*W*C*bs)
% 输出：
%     outPutFeatures： M*(5+nc)或者bs*M*(5+nc)大小
%     ,为bs*M*[x,y,w,h,Pobj,p1,p2,...,pn]大小的形式矩阵，如果是单张图像检测，则输出大小为M*(5+nc)，否则是bs*M*(5+nc),
%     其中，M为检测框的数量，bs为图片数量，nc为训练网络dlnet类别数量，x,y,w,h分别是输入图片上对应的x,y,width,height，Pobj
%     为物体概率，p1,p2,...,pn分别为对应coco.names类别概率
%
% author: cuixingxing
% email:cuixingxing150@gmail.com
% 2020.4.22创建
% 2020.5.2 修改
% 2020.5.13 minor fix
%

persistent dlnet yolovLayerArray netInputSize
if isempty(dlnet)
    
    %% import network and predict
    [layerGraphYolo,hyperParams] = importDarknetWeights(cfg_file,weight_file);
    dlnet = dlnetwork(layerGraphYolo); % 65秒左右
    % analyzeNetwork(layerGraphYolo)% visualize network
    % exportDarkNetNetwork(dlnet,hyperParams,'temp.cfg','temp.weights')
    
     %% get yolo index
    yoloIdx = [];
    for i = 1:length(dlnet.Layers)
        currentLayerType = class(dlnet.Layers(i));
        if strcmpi(currentLayerType,'yoloV3Layer')
            yoloIdx = [yoloIdx;i];
        end
    end
    assert(~isempty(yoloIdx),'输入网络非yolov3/4网络！')

    yolovLayerArray = dlnet.Layers(yoloIdx);
    netInputSize = dlnet.Layers(1).InputSize(1:2);
end

inputSize = [size(image,1),size(image,2)];
scale = inputSize./netInputSize;% [heightScale,widthScale]

img = imresize(im2single(image),netInputSize);% [0,1]数据，保持与训练时候同大小和类型
dlX = dlarray(img,'SSCB');% 大小为h*w*c*bs,注意是否归一化要看与训练时候图像一致
if(canUseGPU())
    dlX = gpuArray(dlX);% 推送到GPU上
end

numsYOLO = length(yolovLayerArray);
outFeatureMaps = cell(numsYOLO,1);
[outFeatureMaps{:}] = predict(dlnet,dlX,'Outputs',dlnet.OutputNames);% h*w*c*bs,matlab输出方式排列
outPutFeatures = [];
for i = 1:numsYOLO
    currentYOLOLayer = yolovLayerArray(i);
    currentFeatureMap = outFeatureMaps{i};
    
    % 由于yolov3Layer类里面predict函数未改变类属性，故重新给属性赋值
    currentYOLOLayer.numY = size(currentFeatureMap,1);
    currentYOLOLayer.numX = size(currentFeatureMap,2);
    currentYOLOLayer.stride = max(currentYOLOLayer.imageSize)./max(currentYOLOLayer.numX,...
        currentYOLOLayer.numY);
    
    % reshape currentFeatureMap到有意义的维度，h*w*c*bs --> h*w*(5+nc)*na*bs
    % --> bs*na*h*w*(5+nc),最终的维度方式与darknet官网兼容
    bs = size(currentFeatureMap,4);
    h = currentYOLOLayer.numY;
    w = currentYOLOLayer.numX;
    na = currentYOLOLayer.nAnchors;
    nc = currentYOLOLayer.classes;
    currentFeatureMap = reshape(currentFeatureMap,h,w,5+nc,na,bs);% h*w*(5+nc)*na*bs
    currentFeatureMap = permute(currentFeatureMap,[5,4,1,2,3]);% bs*na*h*w*(5+nc)
    
    [~,~,yv,xv] = ndgrid(1:bs,1:na,0:h-1,0:w-1);% yv,xv大小都为bs*na*h*w，注意顺序，后面做加法维度标签要对应
    gridXY = cat(5,xv,yv);% 第5维上扩展，大小为bs*na*h*w*2, x,y从1开始的索引
    currentFeatureMap(:,:,:,:,1:2) = sigmoid(currentFeatureMap(:,:,:,:,1:2)) + gridXY; % 大小为bs*na*h*w*2,预测对应xy
    anchor_grid = currentYOLOLayer.anchorsUse/currentYOLOLayer.stride; % 此处anchor_grid大小为na*2
    anchor_grid = reshape(anchor_grid,1,currentYOLOLayer.nAnchors,1,1,2);% 此处anchor_grid大小为1*na*1*1*2，方便下面相乘
    currentFeatureMap(:,:,:,:,3:4) = exp(currentFeatureMap(:,:,:,:,3:4)).*anchor_grid;% 大小为bs*na*h*w*2
    currentFeatureMap(:,:,:,:,1:4) = currentFeatureMap(:,:,:,:,1:4)*currentYOLOLayer.stride; % 预测的bboxes
    currentFeatureMap(:,:,:,:,5:end) = sigmoid(currentFeatureMap(:,:,:,:,5:end)); % 预测的scores

    if currentYOLOLayer.classes == 1
        currentFeatureMap(:,:,:,:,6) = 1;
    end
    currentFeatureMap = reshape(currentFeatureMap,bs,[],5+currentYOLOLayer.classes);% bs*N*(5+nc)
    
    if isempty(outPutFeatures)
        outPutFeatures = currentFeatureMap;
    else
        outPutFeatures = cat(2,outPutFeatures,currentFeatureMap);% bs*M*(5+nc)
    end
end

%% 坐标转换到原始图像上
outPutFeatures = extractdata(outPutFeatures);% bs*M*(5+nc) ,为[x_center,y_center,w,h,Pobj,p1,p2,...,pn]
outPutFeatures(:,:,[1,3]) = outPutFeatures(:,:,[1,3])*scale(2);% x_center,width
outPutFeatures(:,:,[2,4]) = outPutFeatures(:,:,[2,4])*scale(1);% y_center,height
outPutFeatures(:,:,1) = outPutFeatures(:,:,1) -outPutFeatures(:,:,3)/2;%  x
outPutFeatures(:,:,2) = outPutFeatures(:,:,2) -outPutFeatures(:,:,4)/2; % y
outPutFeatures = squeeze(outPutFeatures); % 如果是单张图像检测，则输出大小为M*(5+nc)，否则是bs*M*(5+nc)
if(canUseGPU())
    outPutFeatures = gather(outPutFeatures); % 推送到CPU上
end
end
