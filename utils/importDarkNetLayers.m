function [lgraph,hyperParams,numsNetParams,FLOPs,moduleTypeList,moduleInfoList,layerToModuleIndex] = importDarkNetLayers(cfgfile,cutoffModule)
% importDarkNetLayers 功能：把darknet的cfgfile导出为matlab的lgraph
% 输入：cfgfile, (必选项)字符向量，指定的cfg后缀的模型描述文件名字
%      cutoffModule,(可选项)1*1的正整数，指定导入darknet前cutoffModule个module。cutoffModule是以cfg文件中第一个[convolutional]为0开始的module计数，没有该项输入则导入整个网络
% 输出：lgraph， matlab深度学习模型图
%      hyperParams,结构体，超参配置文件
%      numsNetParams,权重参数个数
%      FLOPs， 模型计算力
%      moduleTypeList,cell array类型，里面每个字符向量存储module的类型
%      moduleInfoList，cell array类型，里面每个struct存储module的信息， 除了里面每个结构体存储cfg中的内容外，还必须存储channels，mapsize两个属性
%      layerToModuleIndex,
%      n*1的向量数组，lgraph中Layers归为module的索引，从1开始，n为Layers的长度大小
% 注意：1、适合2019b版本及以上
%      2、shortcut只支持输入节点数不高于2个
%      特征图输出output Size = (Input Size – ((Filter Size – 1)*Dilation Factor + 1) + 2*Padding)/Stride + 1
% 参考：1、官方文档，Specify Layers of Convolutional Neural Network
%      2、https://www.zhihu.com/question/65305385
% cuixingxing150@gmail.com
% 2019.8.19
% 2019.8.29修改，加入relu6支持
% 2019.9.4修改，由原来的darknet中[net]为0开始的索引改为以cfg文件中第一个非[net]开始的module为0开始的计数的索引
% 2020.4.25修改函数默认输入参数，empty2dLayer加入属性
% 2020.4.29加入[route]层上面多个层连接输入，支持darknet spp模块
% 2020.4.29加入mish激活函数支持
% 2020.6.29加入yolov4-tiny,yolov3-tiny-prn支持

arguments
    cfgfile (1,:) char
    cutoffModule {mustBeNonnegative} = 0 % 默认导入所有的层
end

%% init
numsNetParams = 0;FLOPs = 0;nums_p = 0;FLOPs_perConv = 0;

%% 解析配置cfg文件
fid = fopen(cfgfile,'r');
if fid == -1
  error('Author:Function:OpenFile', 'Cannot open file: %s', cfgfile);
end
cfg = textscan(fid, '%s', 'Delimiter',{'   '});
fclose(fid);
cfg = cfg{1};
TF = startsWith(cfg,'#');
cfg(TF) = [];

%% 网络module和info信息汇集
TF_layer = startsWith(cfg,'[');
moduleTypeList = cfg(TF_layer);
nums_Module = length(moduleTypeList);
moduleInfoList = cell(nums_Module,1);%

%% 读取参数配置文件
indexs = find(TF_layer);
for i = 1:nums_Module
    if i == nums_Module
        moduleInfo = cfg(indexs(i)+1:end,:);
    else
        moduleInfo = cfg(indexs(i)+1:indexs(i+1)-1,:);
    end
    if ~isempty(moduleInfo)
        moduleInfo = strip(split(moduleInfo,'='));
        moduleInfo = reshape(moduleInfo,[],2);
        structArray = cell2struct(moduleInfo, moduleInfo(:,1), 1);
        moduleStruct = structArray(2);
        moduleInfoList{i} = moduleStruct;
    else
        moduleInfoList{i} = [];
    end
end

%% cutoff
if (cutoffModule)
    moduleTypeList(cutoffModule+2:end) = [];
    moduleInfoList(cutoffModule+2:end) = [];
end
nums_Module = length(moduleTypeList);

%% 构建网络结构图
lgraph = layerGraph();hyperParams = struct();
moduleLayers = []; lastModuleNames = cell(nums_Module,1);layerToModuleIndex=[];
yoloIndex = 0;
for i = 1:nums_Module
    currentModuleType = moduleTypeList{i};
    currentModuleInfo = moduleInfoList{i};
    switch currentModuleType
        case '[net]'
            hyperParams = currentModuleInfo;
            if all(isfield(currentModuleInfo,{'height','width','channels'}))
                height = str2double(currentModuleInfo.height);
                width =  str2double(currentModuleInfo.width);
                channels = str2double(currentModuleInfo.channels);
                imageInputSize = [height,width,channels];
                moduleInfoList{i}.channels = channels; % 方便后面计算网络参数个数或者FLOPs
                moduleInfoList{i}.mapSize = [height ,width];% 方便计算FLOPs和最后一层池化大小
            else
                error('[net] require height, width,channels parameters in cfg file!');
            end
            input_layer = imageInputLayer(imageInputSize,'Normalization','none',...
                'Name','input_1');
            moduleLayers = input_layer;
            lgraph = addLayers(lgraph,moduleLayers);
        case '[convolutional]'
            % 添加conv层
            moduleLayers = [];conv_layer = [];bn_layer = [];relu_layer = [];
            nums_p=numsNetParams;% 计算当前module的权重个数
            filterSize = str2double(currentModuleInfo.size);
            numFilters = str2double(currentModuleInfo.filters);
            stride = str2double(currentModuleInfo.stride);
            pad = str2double(currentModuleInfo.pad);
            if stride==1
                pad ='same';
            end
            channels_in = moduleInfoList{i-1}.channels;
            if isfield(currentModuleInfo,'groups')
                numGroups = str2double(currentModuleInfo.groups);
                numFiltersPerGroup_out = numFilters/numGroups;
                conv_layer = groupedConvolution2dLayer(filterSize,numFiltersPerGroup_out,numGroups,...
                    'Name',['dw_conv_',num2str(i)],'Stride',stride,...
                    'Padding',pad);
                numsNetParams = numsNetParams +(filterSize*filterSize*channels_in/numGroups*numFiltersPerGroup_out*numGroups);
                numsNetParams = numsNetParams +numFiltersPerGroup_out*numGroups; % bias
            else
                conv_layer = convolution2dLayer(filterSize,numFilters,'Name',['conv_',num2str(i)],...
                    'Stride',stride,'Padding',pad);
                numsNetParams = numsNetParams +(filterSize*filterSize*channels_in*numFilters);% weights
                numsNetParams = numsNetParams +numFilters; % bias
            end
            moduleInfoList{i}.channels =numFilters;
            if ischar(pad)
                moduleInfoList{i}.mapSize = moduleInfoList{i-1}.mapSize;
            else
                dilationF=1;
                moduleInfoList{i}.mapSize = floor((moduleInfoList{i-1}.mapSize-((filterSize-1)*dilationF +1)+2*pad)/stride+1);
            end
            
            % 添加BN层
            if isfield(currentModuleInfo,'batch_normalize')
                bn_layer = batchNormalizationLayer('Name',['bn_',num2str(i)]);
                numsNetParams = numsNetParams +numFilters*4;% offset,scale,mean,variance
            end
            FLOPs_perConv = prod(moduleInfoList{i}.mapSize)*(numsNetParams-nums_p);
            FLOPs = FLOPs+FLOPs_perConv;

            % 添加relu层
            if strcmp(currentModuleInfo.activation,'relu')
                relu_layer = reluLayer('Name',['relu_',num2str(i)]);
            elseif strcmp(currentModuleInfo.activation,'relu6')
                relu_layer = clippedReluLayer(6,'Name',['clipRelu_',num2str(i)]);
            elseif strcmp(currentModuleInfo.activation,'leaky')
                relu_layer = leakyReluLayer(0.1,'Name',['leaky_',num2str(i)]);
            elseif strcmp(currentModuleInfo.activation,'mish')
                relu_layer = mishLayer(['mish_',num2str(i)]);
            end
            moduleLayers = [conv_layer;bn_layer;relu_layer];
            lgraph = addLayers(lgraph,moduleLayers);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{i-1},moduleLayers(1).Name);
        case '[shortcut]'
            moduleLayers = [];add_layer=[];relu_layer = [];
            connectID = strip(split(currentModuleInfo.from,','));% connectID为cell
            if length(connectID)>2
                error('unsupport more than 2 inputs');
            end
             if length(connectID)==1
                module_idx1 = i-1;
                temp = str2double(connectID);
                module_idx2 = getModuleIdx(i,temp);
            else
                temp1 = str2double(connectID(1));temp2 = str2double(connectID(2));
                module_idx1 = getModuleIdx(i,temp1);
                module_idx2 = getModuleIdx(i,temp2);
             end
             if moduleInfoList{module_idx1}.channels~=moduleInfoList{module_idx2}.channels % yolov3-tiny-prn.cfg  https://github.com/WongKinYiu/PartialResidualNetworks
                 add_layer = prnAdditionLayer(2,['prn_add_',num2str(i)]);
                 moduleInfoList{i}.channels =min(moduleInfoList{module_idx1}.channels,... 
                     moduleInfoList{module_idx2}.channels);
             else
                 add_layer = additionLayer(2,'Name',['add_',num2str(i)]);
                 moduleInfoList{i}.channels =moduleInfoList{module_idx1}.channels;
             end
             moduleInfoList{i}.mapSize = moduleInfoList{module_idx1}.mapSize;
             
            % 添加relu层
            if strcmp(currentModuleInfo.activation,'relu')
                relu_layer = reluLayer('Name',['relu_',num2str(i)]);
            elseif strcmp(currentModuleInfo.activation,'relu6')
                relu_layer = clippedReluLayer(6,'Name',['clipRelu_',num2str(i)]);
            elseif strcmp(currentModuleInfo.activation,'leaky')
                relu_layer = leakyReluLayer('Name',['leaky_',num2str(i)]);
            elseif strcmp(currentModuleInfo.activation,'mish')
                relu_layer = mishLayer(['mish_',num2str(i)]);
            end
            moduleLayers = [add_layer;relu_layer];
            lgraph = addLayers(lgraph,moduleLayers);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{module_idx1},[moduleLayers(1).Name,'/in1']);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{module_idx2},[moduleLayers(1).Name,'/in2']);
        case '[route]' 
            moduleLayers = [];depth_layer = [];
            connectID = strip(split(currentModuleInfo.layers,','));
            numberCon = length(connectID);
            if numberCon==1 % 只有一个连接的时候，可能为empty Layer也可能为yolov4-tiny的分组通道layer，在matlab中自定义sliceLayer
                temp = str2double(connectID);
                module_idx = getModuleIdx(i,temp);
                if all(isfield(currentModuleInfo,{'groups','group_id'}))
                    numGroups = str2double(currentModuleInfo.groups);
                    groupID = str2double(currentModuleInfo.group_id);
                    depth_layer = sliceLayer(['slice_',num2str(i)],module_idx-2,numGroups,groupID+1);
                    moduleInfoList{i}.channels = moduleInfoList{module_idx}.channels/numGroups;
                    moduleInfoList{i}.mapSize = moduleInfoList{module_idx}.mapSize;
                else % 只有一个连接的时候，并且只有'layers'关键字，另一个不是默认上一层，与shortcut层单个值不同,此时为empty Layer
                    depth_layer = empty2dLayer(['empty_',num2str(i)],module_idx-2);
                    moduleInfoList{i}.channels = moduleInfoList{module_idx}.channels;
                    moduleInfoList{i}.mapSize = moduleInfoList{module_idx}.mapSize;
                end
            else % 此时为depthConcatenation Layer
                depth_layer = depthConcatenationLayer(numberCon,'Name',['concat_',num2str(i)]);
                module_idx = ones(numberCon,1);
                channels_add = zeros(numberCon,1);
                for routeii = 1:numberCon % 注意通道连接有顺序要求！不能搞反
                    temp = str2double(connectID(routeii));
                    module_idx(routeii) = getModuleIdx(i,temp);
                    channels_add(routeii) = moduleInfoList{module_idx(routeii)}.channels;
                end
                moduleInfoList{i}.channels = sum(channels_add);
                moduleInfoList{i}.mapSize = moduleInfoList{module_idx(1)}.mapSize;
            end
            
            moduleLayers = [depth_layer;];
            lgraph = addLayers(lgraph,moduleLayers);
            if numberCon == 1
                lgraph = connectLayers(lgraph,...
                    lastModuleNames{module_idx},moduleLayers(1).Name);
            else
                for routeii = 1:numberCon
                     lgraph = connectLayers(lgraph,...
                    lastModuleNames{module_idx(routeii)},[moduleLayers(1).Name,'/in',num2str(routeii)]);
                end
            end
        case '[avgpool]'
            moduleLayers = [];avg_layer = [];
            poolsize = moduleInfoList{i-1}.mapSize;
            pad =0;stride=1;
            if isempty(currentModuleInfo) % 为空时候，自动推断大小,即为上一层特征图大小
                avg_layer = averagePooling2dLayer(poolsize,'Padding',pad,...
                    'Stride',stride,'Name',['avgPool_',num2str(i)]);
            else
                poolsize = str2double(currentModuleInfo.size);
                stride = str2double(currentModuleInfo.stride);
                pad = 'same'; % 确保stride为1时候，特征图大小不变
                 if isfield(currentModuleInfo,'padding')
                    pad = str2double(currentModuleInfo.padding);
                end
                avg_layer = averagePooling2dLayer(poolsize,'Padding',pad,...
                    'Stride',stride,'Name',['avgPool_',num2str(i)]);
            end
            moduleInfoList{i}.channels = moduleInfoList{i-1}.channels;
            if ischar(pad)&&stride==1
                 moduleInfoList{i}.mapSize =  moduleInfoList{i-1}.mapSize;
            elseif ischar(pad)
                moduleInfoList{i}.mapSize = ceil(moduleInfoList{i-1}.mapSize/stride);
            else
                moduleInfoList{i}.mapSize = floor((moduleInfoList{i-1}.mapSize-poolsize+2*pad)/stride+1);
            end    
            
            moduleLayers= avg_layer;
            lgraph = addLayers(lgraph,moduleLayers);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{i-1},moduleLayers(1).Name);
        case '[maxpool]'
            moduleLayers = [];maxp_layer = [];
            poolsize = moduleInfoList{i-1}.mapSize;
            pad =0;stride=1;
            if isempty(currentModuleInfo) % 为空时候，自动推断大小,即为上一层特征图大小
                maxp_layer = maxPooling2dLayer(poolsize,'Padding',pad,...
                    'Stride',stride,'Name',['avgPool_',num2str(i)]);
            else
                poolsize = str2double(currentModuleInfo.size);
                stride = str2double(currentModuleInfo.stride);
                pad = 'same'; % 确保stride为1时候，特征图大小不变
                if isfield(currentModuleInfo,'padding')
                    pad = str2double(currentModuleInfo.padding);
                end
                maxp_layer = maxPooling2dLayer(poolsize,'Padding',pad,...
                    'Stride',stride,'Name',['maxPool_',num2str(i)]);
            end
            moduleInfoList{i}.channels = moduleInfoList{i-1}.channels;
            if ischar(pad)&&stride==1
                 moduleInfoList{i}.mapSize =  moduleInfoList{i-1}.mapSize;
            elseif ischar(pad)
                moduleInfoList{i}.mapSize = ceil(moduleInfoList{i-1}.mapSize/stride);
            else
                moduleInfoList{i}.mapSize = floor((moduleInfoList{i-1}.mapSize-poolsize+2*pad)/stride+1);
            end    
            
            moduleLayers= maxp_layer;
            lgraph = addLayers(lgraph,moduleLayers);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{i-1},moduleLayers(1).Name);
        case '[dropout]'
            moduleLayers = [];drop_layer = [];
            probability = str2double(currentModuleInfo.probability);
            drop_layer = dropoutLayer(probability,'Name',['drop_',num2str(i)]);
            moduleInfoList{i}.channels = moduleInfoList{i-1}.channels;
            moduleInfoList{i}.mapSize = moduleInfoList{i-1}.mapSize;
            
            moduleLayers= drop_layer;
            lgraph = addLayers(lgraph,moduleLayers);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{i-1},moduleLayers(1).Name);
        case '[connected]' % 与普通卷积最大区别是输入大小是否固定，保证全连接层参数可乘;其后没有BN
            moduleLayers = [];connected_layer = [];relu_layer = [];
            output = str2double(currentModuleInfo.output);
            connected_layer = fullyConnectedLayer(output,'Name',['fullyCon_',num2str(i)]);
            moduleInfoList{i}.channels = output;
            moduleInfoList{i}.mapSize = [1,1];
            
            % 添加relu层
            if strcmp(currentModuleInfo.activation,'relu')
                relu_layer = reluLayer('Name',['relu_',num2str(i)]);
            elseif strcmp(currentModuleInfo.activation,'relu6')
                relu_layer = clippedReluLayer(6,'Name',['clipRelu_',num2str(i)]);
            elseif strcmp(currentModuleInfo.activation,'leaky')
                relu_layer = leakyReluLayer(0.1,'Name',['leaky_',num2str(i)]);
            elseif strcmp(currentModuleInfo.activation,'mish')
                relu_layer = mishLayer(['mish_',num2str(i)]);
            end

            moduleLayers= [connected_layer;relu_layer];
            lgraph = addLayers(lgraph,moduleLayers);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{i-1},moduleLayers(1).Name);
        case '[softmax]'
            moduleLayers = [];soft_layer = [];
            soft_layer = softmaxLayer('Name',['softmax_',num2str(i)]);
            moduleInfoList{i}.channels = moduleInfoList{i-1}.channels;
            moduleInfoList{i}.mapSize = moduleInfoList{i-1}.mapSize;
            
            moduleLayers= soft_layer;
            lgraph = addLayers(lgraph,moduleLayers);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{i-1},moduleLayers(1).Name);
        case '[cost]'
            moduleLayers = [];clss_layer = [];
            clss_layer = classificationLayer('Name',['clss_',num2str(i)]);
            moduleInfoList{i}.channels = moduleInfoList{i-1}.channels;
            moduleInfoList{i}.mapSize = moduleInfoList{i-1}.mapSize;
            
            moduleLayers= clss_layer;
            lgraph = addLayers(lgraph,moduleLayers);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{i-1},moduleLayers(1).Name);
        case '[upsample]'   % upsample2dLayer
            moduleLayers = [];up_layer = [];
            stride = str2double(strip(currentModuleInfo.stride));% 整数标量，一般为2
            up_layer = upsample2dLayer(['up2d_',num2str(i)],stride);
            moduleInfoList{i}.channels = moduleInfoList{i-1}.channels;
            moduleInfoList{i}.mapSize = moduleInfoList{i-1}.mapSize*stride;
            
            moduleLayers= up_layer;
            lgraph = addLayers(lgraph,moduleLayers);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{i-1},moduleLayers(1).Name);
        case '[yolo]' % yolov3Layer
            moduleLayers = [];yolov3_layer = [];
            allAnchors = reshape(str2double(strip(split(currentModuleInfo.anchors,','))),2,[])';%[width,height],n*2大小
            mask = reshape(str2double(strip(split(currentModuleInfo.mask,','))),1,[])+1;% 1*m大小,注意mask是从0开始，要加1
            nClasses = str2double(currentModuleInfo.classes);
            imageSize = imageInputSize(1:2); % [imageHeight,imageWidth]
            yoloIndex = yoloIndex + 1;
            yolov3_layer = yolov3Layer(['yolo_v3_id',num2str(yoloIndex)],...
                mask,allAnchors,nClasses,yoloIndex,imageSize);
            moduleInfoList{i}.channels = moduleInfoList{i-1}.channels;
            moduleInfoList{i}.mapSize = moduleInfoList{i-1}.mapSize;
            
            moduleLayers= yolov3_layer;
            lgraph = addLayers(lgraph,moduleLayers);
            lgraph = connectLayers(lgraph,...
                lastModuleNames{i-1},moduleLayers(1).Name);
        otherwise
            error("we currently can't support this layer: "+currentModuleType);
    end
    
    if i==1
        fprintf('This module No:%2d %-16s,have #params:%-10d,FLops:%-12d,feature map size:(%-3d*%-3d),channels in:%-12s,channels out:%-12d\n',...
            i,currentModuleType,numsNetParams-nums_p,FLOPs_perConv,moduleInfoList{i}.mapSize, '-',moduleInfoList{i}.channels);
    else
        fprintf('This module No:%2d %-16s,have #params:%-10d,FLops:%-12d,feature map size:(%-3d*%-3d),channels in:%-12d,channels out:%-12d\n',...
            i,currentModuleType,numsNetParams-nums_p,FLOPs_perConv,moduleInfoList{i}.mapSize,moduleInfoList{i-1}.channels,moduleInfoList{i}.channels);
    end
    nums_p=numsNetParams; 
    FLOPs_perConv = 0;
    lastModuleNames{i} = moduleLayers(end).Name;
    layerToModuleIndex = [layerToModuleIndex;i*ones(length(moduleLayers),1)];
end

    function matlab_module_idx = getModuleIdx(current_matlab_ind,cfg_value)
        % route,或者shortcut层转换为以1为起始索引的标量值
        % 输入：current_matlab_ind，1*1的正整数,读入到当前层module的索引标量(current_matlab_ind中是以[net]以1为起始值),darknet是以第一个非[net]开始的module为0开始的计数的索引
        %      cfg_value，1*1的整数，shortcut层的from值或者route的layers的某一个值
        % 输出：module_idx，连接到上一层module的索引值（正整数,以[net]为起始索引1）
        %
        % cuixingxing150@gmail.com
        % 2019.8.19
        % 2019.9.4修改索引
        if cfg_value<0
            matlab_module_idx = current_matlab_ind+cfg_value;
        else
            matlab_module_idx = 2+cfg_value;
        end
    end % end of getModuleIdx

end % end of importDarknetLayers
