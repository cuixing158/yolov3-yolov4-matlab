%% custom input
addpath('./CustomLayers/','./utils/')
% clear all % 如果更换模型，需重置静态函数(影响性能)，否则可以不用清理
cfg_file = 'cfg/yolov4.cfg';
weight_file = 'weights/yolov4.weights';
throushold = 0.5;
NMS = 0.4;

%% import all classes
fid = fopen('coco.names','r');
names = textscan(fid, '%s', 'Delimiter',{'   '});
fclose(fid);classesNames = categorical(names{1});
RGB = randi(255,length(classesNames),3);

%% 摄像头视频流识别
cap = webcam();
player = vision.DeployableVideoPlayer();
image = cap.snapshot();
step(player, image);

while player.isOpen()
    image = cap.snapshot();
    t1 = tic;
    outFeatures = yolov3v4Predict(cfg_file,weight_file,image);% M*(5+nc) ,为[x,y,w,h,Pobj,p1,p2,...,pn]
    fprintf('预测耗时：%.2f 秒\n',toc(t1));% yolov4大概0.4秒，yolov3大概0.2秒，yolov3-tiny大概0.06秒
    
    %% 阈值过滤+NMS处理
    scores = outFeatures(:,5);
    outFeatures = outFeatures(scores>throushold,:);
    
    allBBoxes = outFeatures(:,1:4);
    allScores = outFeatures(:,5);
    [maxScores,indxs] = max(outFeatures(:,6:end),[],2);
    allScores = allScores.*maxScores;
    allLabels = classesNames(indxs);
    
    % NMS非极大值抑制
    if ~isempty(allBBoxes)
        [bboxes,scores,labels] = selectStrongestBboxMulticlass(allBBoxes,allScores,allLabels,...
            'RatioType','Min','OverlapThreshold',NMS);
        annotations = string(labels) + ": " + string(scores);
        [~,ids] = ismember(labels,classesNames);
        colors = RGB(ids,:);
        image = insertObjectAnnotation(image,...
            'rectangle',bboxes,cellstr(annotations),...
            'Color',colors,...
            'LineWidth',3);
    end
    step(player,image);
end
release(player);



