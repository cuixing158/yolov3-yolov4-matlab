classdef mishLayer < nnet.layer.Layer
    % custom MishLayer layer.激活函数优于relu
    % 参考：https://arxiv.org/abs/1908.08681
    %      《Mish: A Self Regularized Non-Monotonic Neural Activation Function》
    % cuixingxing150@gmail.com
    % 2020.4.29
    %
    properties 
       
    end
    
    methods
        function layer = mishLayer(name)
            % Set layer name.
            layer.Name = name;
            
            % Set layer description.
            layer.Description = "mishLayer activation layer";
            
            layer.Type = 'mishLayer';
        end
        
        function Z = predict(~, X)
            % Z = predict(layer, X) forwards the input data X through the
            % layer and outputs the result Z.
            Z = X.*tanh(log(1+exp(X)));
        end
    end
end