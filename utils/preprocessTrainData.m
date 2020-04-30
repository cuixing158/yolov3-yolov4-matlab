function out = preprocessTrainData(data,networkInputSize,structNamesIDs)
% 功能：对输入数据图像进行预处理，大小一致，加入bbox标签
% 输入：
%     data：bs*3大小的cell array，第一列cell存储图像，第二列cell存储[x,y,w,h]，第三列cell存储classID
%     networkInputSize：输入网络统一大小，[height,width]
%     structNamesIDs: 结构体，类别映射到数字ID
% 输出：
%     out：table类型，1*2大小，第一个为图像数据，第二个为bbox label，形式为[x,y,w,h,classID]
%
% email:cuixingxing150@gmail.com
% 2020.4.22
%

nums = size(data,1);% batchSize大小
XTrain = zeros([networkInputSize,nums],'single');
YTrain = cell(nums,1);% 每个值存储label,x,y,w,h,classID

invalidBboxInd = [];% bbox is invalid and will not be considered in the training set!
for ii = 1:nums
    I = data{ii,1};
    imgSize = size(I);
    
    % Convert an input image with single channel to 3 channels.
    if numel(imgSize) == 1 
        I = repmat(I,1,1,3);
    end
    bboxes = data{ii,2};
    I = im2single(imresize(I,networkInputSize(1:2)));
    scale = networkInputSize(1:2)./imgSize(1:2);
    
    try
        bboxes = bboxresize(bboxes,scale);
    catch
        invalidBboxInd = [invalidBboxInd;ii];
    end
    
    % bbox label
    idxs = zeros(size(bboxes,1),1);
    for jj = 1:size(bboxes,1)
        labels = string(data{ii,3});
        idxs(jj) = structNamesIDs.(labels(jj));
    end
    
    XTrain(:,:,:,ii) = I;
    YTrain{ii} = [bboxes,idxs];
end

% remove invalid bbox and label
XTrain(:,:,:,invalidBboxInd) = [];
YTrain(invalidBboxInd) = [];

out = table({XTrain},{YTrain});
end