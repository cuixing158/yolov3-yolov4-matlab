classdef upsample2dLayer < nnet.layer.Layer
    properties
        size % 标量，一般为2
    end
    
    methods
        function layer = upsample2dLayer(name, size)
            layer.Name = name;
            text = ['[', num2str(size), ' ', num2str(size), '] upsampling for YOLOv3'];
            layer.Description = text;
            layer.Type = 'upsample2dLayer';
            layer.size = size;
        end
        
        function Z = predict(layer, X)
               Z = repelem(X, layer.size, layer.size);
        end
        
        function [dX] = backward( layer, X, ~, dZ, ~ )
            dX = dZ(1:layer.size:end, 1:layer.size:end, :, :);
        end        
    end
end
%% 
% 参考：https://ww2.mathworks.cn/matlabcentral/fileexchange/71277-deep-learning-darknet-importer
% 官方文档：Define Custom Deep Learning Layers
