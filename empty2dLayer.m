classdef empty2dLayer < nnet.layer.Layer
    % 主要想用于导入yolov3中route层的单个跳层连接
    % 2019.9.8
    properties
    end
    
    methods
        function layer = empty2dLayer(name)
            layer.Name = name;
            text = ['[', num2str(1), ' ', num2str(1), '] emptyLayer '];
            layer.Description = text;
            layer.Type = ['empty layer 2d'];
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