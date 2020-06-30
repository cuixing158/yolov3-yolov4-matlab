classdef sliceLayer < nnet.layer.Layer
    % 对应于yolov4-tiny.cfg中的route的group层，用于通道分组
    % https://github.com/AlexeyAB/darknet
    %
    % cuixingxing150@gmail.com
    % 2020.6.29
    properties
        connectID %connectID 是以cfg文件中第一个非[net]开始的module为0开始的计数,用于用于连接前面的层,目的方便exportDarkNetwork函数使用
        groups % 与cfg中的route的group一致
        group_id %  与cfg中的route的group_id一致,group_id在cfg中是从0开始的索引，matlab中为从1开始
    end
    
    methods
        function layer = sliceLayer(name,con,groups,group_id)
            layer.Name = name;
            text = [ num2str(groups), ' groups,group_id: ', num2str(group_id), ' sliceLayer '];
            layer.Description = text;
            layer.Type = 'sliceLayer';
            layer.connectID= con;
            layer.groups= groups;
            layer.group_id= group_id;
            assert(group_id>0,'group_id must great zero! it must start index from 1');
        end
        
        function Z = predict(layer, X) %输出Z保证是4维
            X = reshape(X,[size(X),1]);
            channels = size(X,3);
            deltaChannels = channels/layer.groups;
            selectStart = (layer.group_id-1)*deltaChannels+1;
            selectEnd = layer.group_id*deltaChannels;
            Z = X(:,:,selectStart:selectEnd,:);
        end       
    end
end
