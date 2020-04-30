
addpath('./CustomLayers/','./utils/')
image = imread('./images/dog.jpg');
cfg_file = './cfg/yolov4.cfg';
weight_file = './weights/yolov4.weights';

%% import network and predict
[layerGraphYolo,hyperParams] = importDarknetWeights(cfg_file,weight_file);
dlnet = dlnetwork(layerGraphYolo);
analyzeNetwork(layerGraphYolo)% visualize network
outFeatures = yolov3Predict(dlnet,image);% M*(5+nc) ,为[x,y,w,h,Pobj,p1,p2,...,pn],此函数也适合yolov4进行推理

%% import all classes
fid = fopen('coco.names','r');
names = textscan(fid, '%s', 'Delimiter',{'   '});
fclose(fid);
classesNames = categorical(names{1});
RGB = randi(255,length(classesNames),3);

%% 阈值过滤+NMS处理
throushold = 0.5;
NMS = 0.4;
scores = outFeatures(:,5);
outFeatures = outFeatures(scores>throushold,:);

allBBoxes = outFeatures(:,1:4);
allScores = outFeatures(:,5);
[maxScores,indxs] = max(outFeatures(:,6:end),[],2);
allScores = allScores.*maxScores;
allLabels = classesNames(indxs);

% NMS非极大值抑制
[bboxes,scores,labels] = selectStrongestBboxMulticlass(allBBoxes,allScores,allLabels,...
    'RatioType','Min','OverlapThreshold',NMS);
annotations = string(labels) + ": " + string(scores);
[~,ids] = ismember(labels,classesNames);
colors = RGB(ids,:);
I = insertObjectAnnotation(image,...
    'rectangle',bboxes,cellstr(annotations),...
    'Color',colors,...
    'LineWidth',3);
figure;
imshow(I)


