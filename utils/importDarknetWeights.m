function [lgraph,hyperParams,numsNetParams,FLOPs] = importDarknetWeights(cfgfile,weightsfile,cutoffModule)
% IMPORTDARKNETWEIGHTS 功能：指定导入部分module的darknet模型
% 输入：cfgfile, 指定的cfg后缀的模型描述文件
%       weighfile, 指定的.weights后缀的二进制文件
%       cutoffModule,(可选项)1*1的正整数，(可选项)1*1的正整数，指定导出darknet前cutoffModule个module。以cfg文件中第一个非[net]开始的module为0开始的计数，没有该项则导出整个网络
% 输出：lgraph， matlab深度学习模型计算图
%      hyperParams,结构体，超参配置文件
%      numsReadParams,权重参数个数
%      FLOPs， 模型计算力
% 注意：1、适合2019b版本及以上
%       2、leaky阈值darknet为0.1
%       3、如果某个module中有bn层，则conv的bias为0，因为darknet是这种存储形式
%      4、darknet weights保存顺序依次为BN层offset,scale,mean,variance,Conv层的bias,weights
%      特征图输出output Size = (Input Size – ((Filter Size – 1)*Dilation Factor + 1) + 2*Padding)/Stride + 1
% 参考：1、官方文档，Specify Layers of Convolutional Neural Network
%      2、https://www.zhihu.com/question/65305385
%       3、https://github.com/ultralytics/yolov3/blob/master/models.py
% cuixingxing150@gmail.com
% 2019.8.19
% 2019.9.4修改，由原来的darknet中[net]为0开始的索引改为以cfg文件中第一个非[net]开始的module为0开始的计数的索引
% 2020.4.25修改函数默认输入参数
%
arguments
    cfgfile (1,:) char
    weightsfile (1,:) char
    cutoffModule {mustBeNonnegative} = 0 % 默认导入所有的层
end
    
[lgraph,hyperParams,numsNetParams,FLOPs,...
    moduleTypeList,moduleInfoList,layerToModuleIndex] = importDarkNetLayers(cfgfile);% only weights需要cutoff,layer都导入
assert(length(moduleTypeList)==length(moduleInfoList));

%% 读取权重参数文件
fid_w = fopen(weightsfile,'rb');
if fid_w == -1
  error('Author:Function:OpenFile', 'Cannot open file: %s', weightsfile);
end
header = fread(fid_w, 3, '*int32');
if header(2) > 1
    header2 = fread(fid_w, 1, '*int64'); % int64占用8个字节
else
    header2 = fread(fid_w, 1, '*int32'); % int32占用4个字节
end
fprintf('Major :%d, Minor :%d,Revision :%d,number of images during training:%d,reading params...\n',...
    header(1),header(2),header(3),header2);
weights = fread(fid_w,'*single');
fclose(fid_w);

% numsWeightsParams = numel(weights);
readSize = 1;
numsModule = length(moduleTypeList);
if cutoffModule>0
    numsModule = cutoffModule+1;% [net] plus 1
end

for i = 1:numsModule
    currentModuleType = moduleTypeList{i};
    currentModuleInfo = moduleInfoList{i};
    if strcmp(currentModuleType,'[convolutional]')
        currentModule = lgraph.Layers(i==layerToModuleIndex);
        filterSize = str2double(currentModuleInfo.size);
        numFilters = str2double(currentModuleInfo.filters);
        channels_in = moduleInfoList{i-1}.channels;
            
        if isfield(currentModuleInfo,'batch_normalize')
            % bn bias
            bn_bias = weights(readSize:readSize+numFilters-1);
            bn_bias = reshape(bn_bias,[1,1,numFilters]);
            currentModule(2).Offset = bn_bias;
            readSize = readSize+numFilters;
            % bn weight
            bn_weight = weights(readSize:readSize+numFilters-1);
            bn_weight = reshape(bn_weight,[1,1,numFilters]);
            currentModule(2).Scale = bn_weight;
            readSize = readSize+numFilters;
             % bn trainedMean
            bn_mean = weights(readSize:readSize+numFilters-1);
            bn_mean = reshape(bn_mean,[1,1,numFilters]);
            currentModule(2).TrainedMean = bn_mean;
            readSize = readSize+numFilters;
             % bn trainedVariance
            bn_var = weights(readSize:readSize+numFilters-1);
            bn_var = reshape(bn_var,[1,1,numFilters]);
            if any(bn_var<-0.01)
                error("方差应该大于0！");
            end
            currentModule(2).TrainedVariance = abs(bn_var); % 防止接近于0的数是负数
            readSize = readSize+numFilters;
            % conv bias 为0
            if isfield(currentModuleInfo,'groups')
                numGroups = str2double(currentModuleInfo.groups);
                numFiltersPerGroup_out = numFilters/numGroups;
                currentModule(1).Bias = zeros(1,1,numFiltersPerGroup_out,numGroups,'single');
            else
                currentModule(1).Bias = zeros(1,1,numFilters,'single');
            end
        else
            % load conv bias
            conv_bias = weights(readSize:readSize+numFilters-1);
            if isfield(currentModuleInfo,'groups')
                numGroups = str2double(currentModuleInfo.groups);
                numFiltersPerGroup_out = numFilters/numGroups;
                conv_bias = reshape(conv_bias,1,1,numFiltersPerGroup_out,numGroups);
            else
                conv_bias = reshape(conv_bias,1,1,numFilters);
            end
            currentModule(1).Bias = conv_bias;
            readSize = readSize+numFilters;
        end % end of is bn
        % load conv weights
        if isfield(currentModuleInfo,'groups')
            numGroups = str2double(currentModuleInfo.groups);
            numFiltersPerGroup_out = numFilters/numGroups;
            nums_conv_w = filterSize*filterSize*channels_in/numGroups*numFiltersPerGroup_out*numGroups;
            conv_weights = weights(readSize:readSize+nums_conv_w-1);
            conv_weights = reshape(conv_weights,filterSize,filterSize,channels_in/numGroups,numFiltersPerGroup_out,numGroups);
            conv_weights = permute(conv_weights,[2,1,3,4,5]);
            currentModule(1).Weights = conv_weights;
            readSize = readSize+nums_conv_w;
        else
            nums_conv_w = filterSize*filterSize*channels_in*numFilters;% weights
            conv_weights = weights(readSize:readSize+nums_conv_w-1);
            conv_weights = reshape(conv_weights,filterSize,filterSize,channels_in,numFilters);
            conv_weights = permute(conv_weights,[2,1,3,4]);
            currentModule(1).Weights = conv_weights;
            readSize = readSize+nums_conv_w;
        end % end of load conv weights  
        % 更新参数
        % lgraph.Layers(i==layerToModuleIndex) = currentModule;
        for replaceInd = 1:length(currentModule)
            layerName = currentModule(replaceInd).Name;
            lgraph = replaceLayer(lgraph,layerName,currentModule(replaceInd));
        end
    end % end of module '[convolutional]'
    
    % fullyConnectedLayer 
    if  strcmp(currentModuleType,'[connected]')
        currentModule = lgraph.Layers(i==layerToModuleIndex);
        numFilters = str2double(currentModuleInfo.output);
        % load fc bias
        numBias = numFilters;
        fl_bias = weights(readSize:readSize+numBias-1);
        fl_bias = reshape(fl_bias,numBias,1);
        currentModule(1).Bias = fl_bias;
        readSize = readSize+numBias;
         % load fc weights
        input_all_neurons = prod(moduleInfoList{i-1}.mapSize)*moduleInfoList{i-1}.channels;
        numWeights = numFilters*input_all_neurons; % help fullyConnectedLayer weights
        fl_weights = weights(readSize:readSize+numWeights-1);
        fl_weights = reshape(fl_weights,input_all_neurons,numFilters);
        fl_weights = permute(fl_weights,[2,1]);% fc不需要permute?
        currentModule(1).Weights = fl_weights;
        readSize = readSize+numWeights;
        % 更新参数
        for replaceInd = 1:length(currentModule)
            layerName = currentModule(replaceInd).Name;
            lgraph = replaceLayer(lgraph,layerName,currentModule(replaceInd));
        end
    end % end of module '[connected]'
end % end of nums of module

if isa(lgraph.Layers(end),'nnet.cnn.layer.SoftmaxLayer')
    lastLayerName = lgraph.Layers(end).Name;
    classifyLayer = classificationLayer('Name','classify');
    lgraph = addLayers(lgraph,classifyLayer);
    lgraph = connectLayers(lgraph,lastLayerName,'classify');
end

fprintf('Load parameters succfully!\n')

