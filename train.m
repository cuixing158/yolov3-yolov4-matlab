addpath('./CustomLayers/','./utils/')
%% 1、准备数据，适合yolov3，yolov4，无需VOC-xml格式
% 数据问题参考,满足以下其一即可：
% 1、matlab中标注参考 https://ww2.mathworks.cn/help/vision/ug/get-started-with-the-image-labeler.html?requestedDomain=cn
% 2、外部标注文件导入到matlab参考 https://blog.csdn.net/cuixing001/article/details/77092627
load gTruthPerson.mat % 自己bbox标注文件，格式参考上面链接,最终为table类型，看起来直观
cfg_file = './cfg/yolov3-tiny.cfg';
weight_file = './weights/yolov3-tiny.weights'; %预训练backbone权重，其他类型也OK
annotateImgHeight = 1024; % 自己标注的图像原始高度
annotateImgWeight = 1280; % 自己标注的图像原始宽度

% 类别名字和对应的ID序号
classesNames = gTruth.Properties.VariableNames(2:end);
classIDs = (0:length(classesNames)-1);% 从0开始标注，保持与darknet官网一致
numClasses = length(classesNames);
structNamesIDs = struct();
for i = 1:numClasses
    structNamesIDs.(classesNames{i}) = classIDs(i);
end

% 创建可迭代的数据集
bldsTrain = boxLabelDatastore(gTruth(:, 2:end));
imdsTrain = imageDatastore(gTruth.imageFilename);
miniBatch = 16;
imdsTrain.ReadSize = miniBatch;
bldsTrain.ReadSize = miniBatch;
trainingData = combine(imdsTrain, bldsTrain);

%% 设定超参数,导入训练权重或者导入matlab其他官方预训练权重，这里以darknet中的".weight"二进制权重
[lgModel,hyperParams] = importDarknetWeights(cfg_file,weight_file);
% analyzeNetwork(lgModel);% 可视化导入网络

inputWeight = str2double(hyperParams.width);
inputHeight = str2double(hyperParams.height);
networkInputSize = [inputHeight inputWeight 3];
preprocessedTrainingData = transform(trainingData,@(data)preprocessTrainData(data,networkInputSize,structNamesIDs));
% 预览标注数据
for k = 1:1
    data = read(preprocessedTrainingData);
    I = data{1,1}{1};
    bbox = data{1,2}{1};
    annotatedImage = zeros(size(I),'like',I);
    for i = 1:size(I,4)
        annotatedImage(:,:,:,i) = insertShape(I(:,:,:,i),'Rectangle',bbox{i}(:,1:4));
    end
    annotatedImage = imresize(annotatedImage,2);
    figure
    montage(annotatedImage)
end

numAnchors = 9;
anchorBoxes = estimateAnchorBoxes(trainingData,numAnchors).*[inputHeight/annotateImgHeight,inputWeight/annotateImgWeight];% anchorBoxes是networkInputSize上的大小
area = anchorBoxes(:, 1).*anchorBoxes(:, 2);
[~, idx] = sort(area, 'descend');
anchorBoxes = anchorBoxes(idx, :);
anchorBoxes = round(anchorBoxes);
anchorBoxMasks = {[1,2,3],[4,5,6],[7,8,9]};% 面积大的anchor结合特征图较小的yolov3层，面积小的anchor结合特征图较大的yolov3层


%% 2，搭建darknet网络,加入yolov3Layer
anchorBoxes(:,[2,1]) =  anchorBoxes(:,[1,2]);% anchorBoxes现在是宽高，与darknet官网保持一致
imageSize = lgModel.Layers(1).InputSize(1:2);
arc = 'default';
yoloModule1 = [convolution2dLayer(1,length(anchorBoxMasks{1})*(5+numClasses),'Name','yoloconv1');
    yolov3Layer('yolov3layer1',anchorBoxes(anchorBoxMasks{1},:),numClasses,1,imageSize,arc)];
yoloModule2 = [convolution2dLayer(1,length(anchorBoxMasks{2})*(5+numClasses),'Name','yoloconv2');
    yolov3Layer('yolov3layer2',anchorBoxes(anchorBoxMasks{2},:),numClasses,2,imageSize,arc)];
lgModel = removeLayers(lgModel,{'yolo_v3_id1','yolo_v3_id2'});
lgModel = replaceLayer(lgModel,'conv_17',yoloModule1);
lgModel = replaceLayer(lgModel,'conv_24',yoloModule2);

analyzeNetwork(lgModel);
yoloLayerNumber = [36,47];% 注意！！！！！yolov3或者yolov4层在layers数组中的位置，看模型图得出！！！！！
model = dlnetwork(lgModel);

%% 3，for loop循环迭代更新模型
% 训练选项
numIterations = 2000;
learningRate = 0.001;
warmupPeriod = 1000;
l2Regularization = 0.0005;
penaltyThreshold = 0.5;
velocity = [];

executionEnvironment = "auto";
figure;
ax1 = subplot(211);
ax2 = subplot(212);
lossPlotter = animatedline(ax1);
learningRatePlotter = animatedline(ax2);

nEpochs = 10;
allIteration = 1;
for numEpoch = 1:nEpochs
    reset(preprocessedTrainingData);% Reset datastore.
    iteration = 1;
    while hasdata(preprocessedTrainingData)
        t_start = tic;
        % Custom training loop.
        % Read batch of data and create batch of images and
        % ground truths.
        outDataTable = read(preprocessedTrainingData);
        XTrain = outDataTable{1,1}{1};
        YTrain = outDataTable{1,2}{1};
        if isempty(YTrain)
            continue;
        end
        
        % Convert mini-batch of data to dlarray.
        XTrain = dlarray(single(XTrain),'SSCB');
        
        % If training on a GPU, then convert data to gpuArray.
        if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
            XTrain = gpuArray(XTrain);
        end
        
        % Evaluate the model gradients and loss using dlfeval and the
        % modelGradients function.
        [gradients,loss,state] = dlfeval(@modelGradients, model, XTrain, YTrain,yoloLayerNumber);
        
        % Apply L2 regularization.
        gradients = dlupdate(@(g,w) g + l2Regularization*w, gradients, model.Learnables);
        
        % Update the network learnable parameters using the SGDM optimizer.
        [model, velocity] = sgdmupdate(model, gradients, velocity, learningRate);
        
        % Update the state parameters of dlnetwork.
        model.State = state;
        
        % save model
        if (mod(numEpoch,5)==0)&&(iteration==1) % 设置每5个epoch保存下权重
            timeStr = datestr(now,'yyyy_mm_dd_HH_MM_SS');
            matlabModel = fullfile('./save',[timeStr,'.mat']);
            save(matlabModel,'model');
            
            cfgFile = fullfile('./cfg',[timeStr,'.cfg']);
            darknetModel = fullfile('./weights',[timeStr,'.weights']);
            exportDarkNetNetwork(model,hyperParams,cfgFile,darknetModel);
        end
        fprintf('[%d][%d/%d]\t BatchTime:%.2f\n\n',numEpoch,iteration,...
            floor(numpartitions(preprocessedTrainingData)/miniBatch),toc(t_start));
        
        % Update training plot with new points.
        addpoints(lossPlotter, allIteration, double(gather(extractdata(loss))));
        %             addpoints(learningRatePlotter, iteration, currentLR);
        iteration = iteration +1;
        allIteration = allIteration+1;
        drawnow
    end
end

%% yolov3/yolov4 损失函数
function [gradients, totalLoss, state] = modelGradients(net, XTrain, YTrain,yoloLayerNumber)
% 功能：计算模型梯度，求取损失
% allYoloLayers = net.Layers(yoloLayerNumber);
yolov3layerNames = net.OutputNames;
outFeatureMaps = cell(size(yolov3layerNames));
[outFeatureMaps{:},state] = forward(net,XTrain,'Outputs',yolov3layerNames);
boxLoss = dlarray(0);
objLoss = dlarray(0);
clsLoss = dlarray(0);
for i = 1:length(outFeatureMaps)
    currentYOLOV3Layer = net.Layers(yoloLayerNumber(i));
    currentFeatureMap = outFeatureMaps{i};
    
    % 由于yolov3Layer类里面predict函数未改变类属性，故重新给属性赋值
    currentYOLOV3Layer.numY = size(currentFeatureMap,1);
    currentYOLOV3Layer.numX = size(currentFeatureMap,2);
    currentYOLOV3Layer.stride = max(currentYOLOV3Layer.imageSize)./max(currentYOLOV3Layer.numX,...
        currentYOLOV3Layer.numY);
    
    % reshape currentFeatureMap到有意义的维度，h*w*c*bs --> h*w*(5+nc)*na*bs
    % --> bs*na*h*w*(5+nc),最终的维度方式与darknet官网兼容
    bs = size(currentFeatureMap,4);
    h = currentYOLOV3Layer.numY;
    w = currentYOLOV3Layer.numX;
    na = currentYOLOV3Layer.nAnchors;
    nc = currentYOLOV3Layer.nClasses;
    arc = currentYOLOV3Layer.arc;
    currentFeatureMap = reshape(currentFeatureMap,h,w,5+nc,na,bs);% h*w*(5+nc)*na*bs
    currentFeatureMap = permute(currentFeatureMap,[5,4,1,2,3]);% bs*na*h*w*(5+nc)
    
    % 构建目标值
    [tcls,tbox,indices,anchor_grids] = buildTargets(currentYOLOV3Layer,YTrain);
    N = size(tcls,1);% N<=YTrain中所有检测框的数量，其代表有效的数量
    tobj = zeros(N,1);
    featuresCh = zeros(N,(5+nc),'like',currentFeatureMap);
    if N
        b = indices(:,1); % N*1
        a = indices(:,2); % N*1
        gj = indices(:,3); % N*1
        gi = indices(:,4); % N*1
        for idx = 1:N
            featuresChannels = currentFeatureMap(b(idx),a(idx),gj(idx),gi(idx),:);% 1*1*1*1*(5+nc)
            featuresChannels = squeeze(featuresChannels);%(5+nc)*1
            featuresChannels = featuresChannels';%1*(5+nc)
            featuresCh(idx,:) = featuresChannels; % N*(5+nc)
            tobj(idx) = 1.0;
        end
        
        % mse or GIoU loss 
        predictXY = sigmoid(featuresCh(:,1:2)); % 大小为N*2,预测对应xy
        predictWH = exp(featuresCh(:,3:4)).*anchor_grids;% 大小为N*2
        predictBboxs = cat(2,predictXY,predictWH);% 大小为N*4
        isUseGIOU = 0;
        if isUseGIOU
            giouRatio = getGIOU(predictBboxs,tbox);%梯度需要计算，然而反向传播非常耗时
            boxLoss = boxLoss+mean(1-giouRatio,'all');
        else
            boxLoss = mse(predictBboxs,tbox,'DataFormat','BC');
        end
        
        if strcmpi(arc,'default')&&(nc>1)
            tcls_ = zeros('like',featuresCh(:,6:end));
            for idx = 1:N
                tcls_(idx,tcls+1) = 1.0;% 确保类别标签是从0开始标注的索引，否则这里会超出维度
            end
            clsLoss = clsLoss + crossentropy(sigmoid(featuresCh(:,6:end)),tcls_,...
                'DataFormat','BC',...
                'TargetCategories','independent');
        end
    else
        
    end
    
    if strcmpi(arc,'default')
        if N
            objLoss = objLoss+crossentropy(sigmoid(featuresCh(:,5)),tobj,...
                'DataFormat','BC',...
                'TargetCategories','independent');
        end
    elseif strcmpi(arc,'uCE')||strcmpi(arc,'uBCE') % obj和class当成一个类别统一计算损失
        if N
            b = indices(:,1); % N*1
            a = indices(:,2); % N*1
            gj = indices(:,3); % N*1
            gi = indices(:,4); % N*1
            tcls_ = zeros('like',featuresCh(:,5:end));
            for idx = 1:N
                featuresChannels = currentFeatureMap(b(idx),a(idx),gj(idx),gi(idx),:);% 1*1*1*1*(5+nc)
                featuresChannels = squeeze(featuresChannels);%(5+nc)*1
                featuresChannels = featuresChannels';%1*(5+nc)
                featuresCh(idx,:) = featuresChannels; % N*(5+nc)
                tcls_(idx,tcls+1) = 1.0;
            end
            clsLoss = clsLoss + crossentropy(featuresCh(:,5:end),tcls_,...
                'DataFormat','BC',...
                'TargetCategories','independent');
        end 
    end
end
    totalLoss = boxLoss+objLoss+clsLoss;
    fprintf('boxLoss:%.2f, objLoss:%.2f, clsLoss:%.2f, totalLoss:%.2f\n',...
        boxLoss,objLoss,clsLoss,totalLoss);
    
    % Compute gradients of learnables with regard to loss.
    gradients = dlgradient(totalLoss, net.Learnables);
end

function [tcls,tbox,indices,anchor_grids] = buildTargets(currentYOLOV3Layer,YTrain)
% 功能：构建目标值
% 输入：
%     currentYOLOV3Layer:网络中yolo输出层之一
%     YTrain:网络目标值，bs*1大小的cell类型，每个cell下包含Mi*[x,y,width,height,classID]大小的矩阵,Mi为第i张图片含有目标的检测数量,
%            注意其存储的坐标值是相对网络输入图像上的坐标，并无归一化
% 输出：
%      tcls:目标真实类别classID，N*1大小，每一项存储classID,其中N<=sum(Mi),只输出有效数量的类别N
%      tbox:目标的boundingBox，存储真实目标在特征图上的位置（除去x,y整数部分,保留小数），N*4大小,每项形式为[Xdecimal,Ydecimal,gw,gh]
%      indices:目标检测框在高维数组中的位置，N*4大小，每一项存储检测框的位置，其形式为[bs,na,gy,gx],他们都是从1开始的索引，与Python不同
%      anchor_grids：所用有效的在特征图上的anchor，N*2大小，每项形式为[anchor_w,anchor_h]
% 注意：
%    此函数是核心，用于产生各个yolov3损失类型的目标，输出每个参数维度都有意义，顺序要保持一致，总的高维顺序为bs*na*h*w*(5+nc),此顺序为darknet
%    官网的顺序一致，非matlab官方一致
%
% author:cuixingxing
% emalil:cuixingxing150@email.com
% 2020.4.25
%
h = currentYOLOV3Layer.numY;
w = currentYOLOV3Layer.numX;
stride = currentYOLOV3Layer.stride;
bs = size(YTrain,1);

% 把targets转换成nt*[imageIDs,classIDs,gx,gy,gw,gh]二维矩阵
scalex = w/currentYOLOV3Layer.imageSize(2);
scaley = h/currentYOLOV3Layer.imageSize(1);
gwh = currentYOLOV3Layer.anchors/stride; % 此处anchor_grids大小为na*2
targets = cat(1,YTrain{:}); % nt*5大小矩阵,nt为该bach下目标检测框数量
output = cellfun(@(x)size(x,1),YTrain);
imageIDs = repelem((1:bs)',output);
classIDs =  targets(:,5);
targets = [imageIDs,classIDs,targets(:,1:end-1)];% nt*6大小，[imageIDs,classIDs,x,y,w,h]

% 计算目标检测框在特征图上的大小
targets(:,[3,5]) = targets(:,[3,5]).*scalex;% gx,gw
targets(:,[4,6]) = targets(:,[4,6]).*scaley;% nt*6大小，[imageIDs,classIDs,gx,gy,gw,gh]

% 分别获取每个anchor每个bbox的target
if ~isempty(targets)
    iou = getMaxIOUPredictedWithGroundTruth(gwh,targets(:,5:6));
    iouThresh = 0.2;
    
    reject = true;
    if reject
        iou(iou<iouThresh) = 0;
    end
    
    use_best_anchor = false;
    if use_best_anchor
        [iou,anchorsID] = max(iou,[],1);
        [~,targetsID] = find(iou);
        targets = targets(targetsID,:)
    else % use all anchors
        [anchorsID,targetsID] = find(iou);
        targets = targets(targetsID,:); % N*6 ,[imageIDs,classIDs,gx,gy,gw,gh]
    end
    anchorsID = anchorsID(:); % N*1
end

gxy = targets(:,3:4);
targets(:,3:4) = gxy -floor(gxy);

% 返回targets值
tcls = targets(:,2);
if ~isempty(tcls)
    assert(max(tcls)<=currentYOLOV3Layer.nClasses,'Target classes exceed model classes!');
end
tbox = targets(:,3:6);
xyPos = ceil(gxy);% 取ceil是因为matlab数组索引从1开始
indices = [targets(:,1),anchorsID,xyPos(:,2),xyPos(:,1)];
anchor_grids = gwh(anchorsID,:);
end

function iou = getMaxIOUPredictedWithGroundTruth(gwh,truth)
% getMaxIOUPredictedWithGroundTruth computes the maximum intersection over
%  union scores for every pair of predictions and ground-truth boxes.
% 输入：
%      gwh: 特征图上的anchor,大小为na*2
%      truth:特征图上的目标真实检测框，只含有宽高,大小为nt*2
% 输出：
%     iou:大小为na*nt,每一个值代表第i个anchor与第j个真值之间的交并比
%
% author:cuixingxing
% emalil:cuixingxing150@email.com
% 2020.4.25
%
bboxA = [ones(size(gwh)),gwh];
bboxB = [ones(size(truth)),truth];
iou = bboxOverlapRatio(bboxA,bboxB);
end