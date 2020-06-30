classdef prnAdditionLayer < nnet.layer.Layer
    % 对应于yolov3-tiny-prn.cfg中的shortcut层，通道部分相加
    % https://github.com/WongKinYiu/PartialResidualNetworks
    %
    % cuixingxing150@gmail.com
    % 2020.6.29
    methods
        function layer = prnAdditionLayer(numInputs,name) 
            % layer = prnAdditionLayer(numInputs,name) creates a
            % prn addition layer and specifies the number of inputs
            % and the layer name.
            
            % Set number of inputs.
            layer.NumInputs = numInputs;

            % Set layer name.
            layer.Name = name;

            % Set layer description.
            layer.Description = "Prn addition of " + numInputs +  ... 
                " inputs";
        end
        
        function Z = predict(layer, varargin)
            % Z = predict(layer, X1, ..., Xn) forwards the input data X1,
            % ..., Xn through the layer and outputs the result Z.
            X = varargin;
            
            % Initialize output
            [minChannels,ind] = min(cellfun(@(x)min(size(x,3)),X));
            X1 = X{1};
            [h,w,~,n] = size(X1);
            Z = zeros([h,w,minChannels,n],'like',X1);
            
            % prn addition
            for i = 1:layer.NumInputs
                item = X{i};
                if i ~= ind
                    startInd = 1;
                    endInd = minChannels;
                    item = item(:,:,startInd:endInd,:);
                end
                Z = Z + item;
            end
        end
    end
end
