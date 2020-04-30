classdef empty2dLayer < nnet.layer.Layer
    % 主要想用于导入yolov3中route层的单个跳层连接
    % 2019.9.8
    % 2020.4.29加入属性connectID，用于连接前面的层
    properties
        connectID %connectID 是以cfg文件中第一个非[net]开始的module为0开始的计数
    end
    
    methods
        function layer = empty2dLayer(name,con)
            layer.Name = name;
            text = ['[', num2str(1), ' ', num2str(1), '] emptyLayer '];
            layer.Description = text;
            layer.Type = 'empty2dLayer';
            layer.connectID= con;
        end
        
        function Z = predict(layer, X)
               Z = X;
        end
        
        function [dX] = backward( layer, X, ~, dZ, ~ )
            dX = dZ;
        end        
    end
end
%% 
% 参考：https://ww2.mathworks.cn/matlabcentral/fileexchange/71277-deep-learning-darknet-importer