function exportDarkNetNetwork(net,hyperParams,cfgfileName,weightfileName,cutoffModule)
% EXPORTDARKETNetNetwork 功能：把matlab深度学习模型导出为darknet的模型weights文件
% 输入：net， matlab深度学习模型,SeriesNetwork,DAGNetwork,dlnetwork之一类型
%      hyperParams,结构体，超参配置文件,对应cfg文件中的[net]参数
%      cutoffModule,(可选项)1*1的正整数，指定导出darknet前cutoffModule个module。cutoffModule是以cfg文件中第一个[convolutional]为0开始的module计数，没有该项则导出整个网络
% 输出：
%      cfgfile, 指定的cfg后缀的模型描述文件
%      weightfile,制定对应的weights权重文件
%
% 注意：1、relu6用relu激活函数代替，因为clip relu不知道darknet是否实现
%      2、matlab中module以[net]为1开始计数，而darknet中cfg以[convolutional]第一个为0开始计数
% cuixingxing150@gmail.com
% 2019.8.22
% 2019.8.29修改，支持导出relu6
% 2019.9.4修改，由原来的darknet中[net]为0开始的索引改为以cfg文件中第一个非[net]开始的module为0开始的计数的索引
% 2020.4.28修改，加入[yolo]、[upsample]、[route]支持;输入参数限定
% 2020.4.29加入mishLayer导出层支持
% 2020.6.29 加入导出yolov4-tiny支持
%

arguments
    net (1,1)  
    hyperParams (1,1) struct
    cfgfileName (1,:) char
    weightfileName (1,:) char
    cutoffModule {mustBeNonnegative} = 0 % 默认导出所有的层
end

%% init
moduleTypeList = []; % cell array,每个cell存储字符向量的模块类型，如'[convolutional]'
moduleInfoList = []; % cell array,每个cell存储结构图的模块信息
layerToModuleIndex = []; % 正整数n*1的vector,每个值代表matlab中从layers映射到module的类别

%% 1、解析net中模块
module_idx = 1;
layerNames = [];% 字符向量n*1的vector,每个值存储每个层的名字，后面shortcut,route要用到
numsLayers = length(net.Layers);
for i = 1:numsLayers
    is_new_module = true;st = struct();
    layerNames = [layerNames;{net.Layers(i).Name}];
    currentLayerType = class(net.Layers(i));
    if strcmpi(currentLayerType,'nnet.cnn.layer.ImageInputLayer')
        moduleTypeList = [moduleTypeList;{'[net]'}];
        st = hyperParams;
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.Convolution2DLayer')
        moduleTypeList = [moduleTypeList;{'[convolutional]'}];
        layer = net.Layers(i);
        st = struct('filters',sum(layer.NumFilters),...
            'size',layer.FilterSize(1),...
            'pad',floor(layer.FilterSize(1)/2),...% 2020.5.7修改
            'stride',layer.Stride(1),...
            'activation','linear');
    elseif strcmpi(currentLayerType, 'nnet.cnn.layer.GroupedConvolution2DLayer')
        moduleTypeList = [moduleTypeList;{'[convolutional]'}];
        layer = net.Layers(i);
        st = struct('groups',layer.NumGroups,...
            'filters',layer.NumGroups*layer.NumFiltersPerGroup,...
            'size',layer.FilterSize(1),...
            'pad',floor(layer.FilterSize(1)/2),...% 2020.5.7修改
            'stride',layer.Stride(1),...
            'activation','linear');
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.FullyConnectedLayer')
        moduleTypeList = [moduleTypeList;{'[connected]'}];
        layer = net.Layers(i);
        st = struct('output',layer.OutputSize,...
            'activation','linear');
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.BatchNormalizationLayer')
        module_idx = module_idx-1;
        moduleInfoList{end}.batch_normalize = 1;
        is_new_module = false;
    elseif  strcmpi(currentLayerType,'nnet.cnn.layer.ReLULayer')
        module_idx = module_idx-1;
        moduleInfoList{end}.activation = 'relu';
        is_new_module = false;
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.LeakyReLULayer')
        module_idx = module_idx-1;
        moduleInfoList{end}.activation = 'leaky';
        is_new_module = false;
    elseif strcmpi(currentLayerType,'mishLayer') % 2020.4.29新加入
        module_idx = module_idx-1;
        moduleInfoList{end}.activation = 'mish';
        is_new_module = false;
    elseif strcmpi(currentLayerType,'nnet.onnx.layer.ClipLayer')%当作阈值为6导出
        module_idx = module_idx-1;
        moduleInfoList{end}.activation = 'relu6'; %实际上类似于matlab的clippedReluLayer,6
        is_new_module = false;
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.ClippedReLULayer') %当作阈值为6导出
        module_idx = module_idx-1;
        moduleInfoList{end}.activation = 'relu6'; %实际上类似于matlab的clippedReluLayer,6
        is_new_module = false;
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.MaxPooling2DLayer')
        moduleTypeList = [moduleTypeList;{'[maxpool]'}];
        layer = net.Layers(i);
        if i==numsLayers-3||i==numsLayers-2 % 最后一层，留作自动推断特征图大小
            st = struct();
        else
            if strcmp(layer.PaddingMode,'manual')
                st = struct('size',layer.PoolSize(1),...
                    'stride',layer.Stride(1),...
                    'padding',layer.PaddingSize(1));
            else
                st = struct('size',layer.PoolSize(1),...
                    'stride',layer.Stride(1));
            end
        end
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.GlobalMaxPooling2DLayer')
        moduleTypeList = [moduleTypeList;{'[maxpool]'}];
        if i==numsLayers-3||i==numsLayers-2% 最后一层，留作自动推断特征图大小
            st = struct();
        end
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.AveragePooling2DLayer')
        moduleTypeList = [moduleTypeList;{'[avgpool]'}];
        layer = net.Layers(i);
        if i==numsLayers-3||i==numsLayers-2% 最后一层，留作自动推断特征图大小
            st = struct();
        else
            if strcmp(layer.PaddingMode,'manual')
                st = struct('size',layer.PoolSize(1),...
                    'stride',layer.Stride(1),...
                    'padding',layer.PaddingSize(1));
            else
                st = struct('size',layer.PoolSize(1),...
                    'stride',layer.Stride(1));
            end
        end
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.GlobalAveragePooling2DLayer')
        moduleTypeList = [moduleTypeList;{'[avgpool]'}];
        if i==numsLayers-3||i==numsLayers-2% 最后一层，留作自动推断特征图大小
            st = struct();
        end
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.SoftmaxLayer')
        moduleTypeList = [moduleTypeList;{'[softmax]'}];
        st = struct('groups',1);
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.AdditionLayer')
        moduleTypeList = [moduleTypeList;{'[shortcut]'}];
        st = struct('from',[],'activation','linear');
        layer_name = layerNames{i};
        index_Dlogical = startsWith(net.Connections.Destination,[layer_name,'/']);
        source = net.Connections.Source(index_Dlogical);
        index_Slogical = contains(layerNames(1:end-1),source);
        st.from = layerToModuleIndex(index_Slogical)-2; % -2 darknet module 是以第一个非[net]开始的module为0的计数
        st.from = num2str(min(st.from)); % 2019.8.29修改
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.DepthConcatenationLayer')
        moduleTypeList = [moduleTypeList;{'[route]'}];
        st = struct('layers',[]);
        layer_name = layerNames{i};
        index_Dlogical = startsWith(net.Connections.Destination,[layer_name,'/']);
        source = net.Connections.Source(index_Dlogical);
        index_Slogical = ismember(layerNames(1:end-1),source); % 2020.4.29日contains改为ismember
        st.layers = layerToModuleIndex(index_Slogical)-2; % -2 darknet module 是以第一个非[net]开始的module为0的计数
        st.layers = join(string(flip(st.layers)),','); % 注意route多个层连接顺序，先连接最近的层，再连接较远的层
    elseif strcmpi(currentLayerType,'nnet.cnn.layer.DropoutLayer')
        moduleTypeList = [moduleTypeList;{'[dropout]'}];
        layer = net.Layers(i);
        st = struct('probability',layer.Probability);
    elseif strcmpi(currentLayerType,'upsample2dLayer')
        moduleTypeList = [moduleTypeList;{'[upsample]'}];
        layer = net.Layers(i);
        st = struct('stride',layer.size);
    elseif strcmpi(currentLayerType,'empty2dLayer')
         moduleTypeList = [moduleTypeList;{'[route]'}];
        layer = net.Layers(i);
        st = struct('layers',[]);
        st.layers = layer.connectID; 
    elseif strcmpi(currentLayerType,'sliceLayer')
        moduleTypeList = [moduleTypeList;{'[route]'}];
        layer = net.Layers(i);
        st = struct('layers',layer.connectID,...
            'groups',layer.groups,...
            'group_id',layer.group_id-1);
    elseif strcmpi(currentLayerType,'yoloV3Layer')
        moduleTypeList = [moduleTypeList;{'[yolo]'}];
        layer = net.Layers(i);
        anchors = layer.anchors';% 2*n
        anchors = reshape(anchors(:),1,[]); % 1*m
        st = struct('mask',join(string(layer.mask-1),','),...
            'anchors',join(string(anchors),','),...
            'classes',num2str(layer.classes),...
            'num',num2str(layer.num),...
            'jitter',num2str(layer.jitter),...
            'ignore_thresh',num2str(layer.ignore_thresh),...
            'truth_thresh',num2str(layer.truth_thresh),...
            'random',num2str(layer.random));
    elseif strcmpi(currentLayerType, 'nnet.cnn.layer.ClassificationOutputLayer')
        continue;
    else
        moduleTypeList = [moduleTypeList;{'[unknow]'}];% 这里需要手动在cfg文件中修改
        st = struct('error',['unsupported this type:',currentLayerType,...
            ',you should manully modify it!']);
    end
    % 更新
    if is_new_module
        moduleInfoList = [moduleInfoList;{st}];
    end
    layerToModuleIndex = [layerToModuleIndex;module_idx];
    module_idx = module_idx+1;
end % 终止解析

%% cutoff
if cutoffModule
    moduleTypeList(cutoffModule+2:end) = [];
    moduleInfoList(cutoffModule+2:end) = [];
end

%% 2、写入cfg模型描述文件
assert(length(moduleTypeList)==length(moduleInfoList));
nums_module = length(moduleTypeList);
fid_cfg = fopen(cfgfileName,'w');
for i = 1:nums_module
    currentModuleType = moduleTypeList{i};% currentModuleType是字符向量类型
    currentModuleInfo = moduleInfoList{i}; % currentModuleInfo是struct类型
    % 逐个module参数写入
    if i==1
        fprintf(fid_cfg,'%s\n','# This file is automatically generated by MATLAB and may require you to modify it manually');% 注释部分
        fprintf(fid_cfg,'%s\n',currentModuleType);% module的名字
    else
        fprintf(fid_cfg,'%s\n',['# darknet module ID:',num2str(i-2)]); %cfg中正式部分
        fprintf(fid_cfg,'%s\n',currentModuleType);% module的名字
    end
    fields = sort(fieldnames(currentModuleInfo));
    if (~isempty(fields)) && contains(fields{1},'activation')
        fields = circshift(fields,-1);% 左移一位,即移到最后
    end
    for j = 1:length(fields) %写入module的结构体信息
        fieldname = fields{j};
        fieldvalue = currentModuleInfo.(fieldname);
        fprintf(fid_cfg,'%s=%s\n',fieldname,num2str(fieldvalue));% module的名字
    end
    fprintf(fid_cfg,'\n');
end
fclose(fid_cfg);

%% 3、保存weights权重
fid_weight = fopen(weightfileName,'wb');
fwrite(fid_weight,[0,2,5],'int32');% version
fwrite(fid_weight,0,'int64'); % number images in train
nums_module = length(moduleTypeList);
for module_index = 1:nums_module
    currentModuleType = moduleTypeList{module_index};% 字符向量
    currentModuleInfo = moduleInfoList{module_index}; % struct
    currentModule = net.Layers(module_index == layerToModuleIndex);
    if strcmp(currentModuleType,'[convolutional]')||strcmp(currentModuleType,'[connected]')
        conv_layer = currentModule(1);
        % 如果该module有BN，首先存储BN的参数
        if isfield(currentModuleInfo,'batch_normalize') % darknet一个弊端，丢弃了conv bias的参数
            bn_layer = currentModule(2);
            bn_bias = bn_layer.Offset;
            fwrite(fid_weight,bn_bias(:),'single');
            bn_weights = bn_layer.Scale;
            fwrite(fid_weight,bn_weights(:),'single');
            bn_mean = bn_layer.TrainedMean;
            fwrite(fid_weight,bn_mean(:),'single');
            bn_var = bn_layer.TrainedVariance;
            fwrite(fid_weight,bn_var(:),'single');
        else
            % conv bias
            conv_bias = conv_layer.Bias;
            conv_bias = permute(conv_bias,[2,1,3,4]);% 支持 groupedConvolution2dLayer
            fwrite(fid_weight,conv_bias(:),'single');
        end
        % conv weights
        conv_weights = conv_layer.Weights;
        conv_weights = permute(conv_weights,[2,1,3,4,5]);% 支持 groupedConvolution2dLayer
        fwrite(fid_weight,conv_weights(:),'single');
    end
end
fclose(fid_weight);

