classdef yolov3Layer < nnet.layer.Layer
    % 该自定义类仍有很大不灵活性，表现在1、predict函数输出不能自由控制；2、类属性在predict函数里面无法改变；3、训练阶段forward函数不执行
    % 4、该类无法自定义其他普通成员函数。
    % 参考官方文档：Define Custom Deep Learning Layers
    properties
        anchors %当前yolov3层的anchors，n*2大小，[width, height]，宽高为相对输入原图大小,注意这个与matlab官网estimateAnchorBoxes的宽高顺序相反
%         anchorWH     % 当前yolov3层的特征上的anchors，n*2大小，[width, height]，宽高为相对特征图大小，anchorWH = anchors./strides
        nAnchors % 当前yolov3层使用anchors的数量，一般为3
        nClasses   % 目标检测总共类别数量
        numX          % 传入到该yolov3层特征图的宽，像素
        numY          % 传入到该yolov3层特征图的宽，像素
        imageSize     % 网络输入图像大小，[imageHeight,imageWidth]
        arc           % 方式,取'default'、'uCE'、'uBCE'中的一种
        stride       % 传入到该yolov3层特征下采样率，[w,h],如输入网络大小是416*416宽乘高，该特征层为13*13，则stride = 32
    end
    
    methods
        function layer = yolov3Layer(name, anchors,nClasses,yoloIndex,imageSize,arc)
            % 输入参数：
            % name: 字符向量或字符串，层名字
            % anchor: n*2大小，[width, height]，宽高相对输入原图大小
            % nClasses： 1*1 标量，检测的类别数量
            % imageSize：[imageHeight,imageWidth],输入网络的图像大小
            % yoloIndex：1*1标量，从1开始的索引，输出yolo检测层的序号
            % arc:字符向量或字符串，计算损失的方式
            %
            % cuixingxing150@gmail.com
            % 2020.4.20
            %
            
            layer.Name = name;
            text = ['all number classes:', num2str(nClasses),...
                ', anchor box:',mat2str(round(anchors)),...
                ', yoloLayerID:',num2str(yoloIndex),...
                ', arc:',arc];
            layer.Description = text;
            layer.Type = 'yoloV3Layer';
            layer.numY = 1;
            layer.numX = 1;
            layer.stride =1;
            
            layer.anchors = anchors;
            layer.nAnchors = size(anchors,1);
            layer.nClasses = nClasses;
            layer.imageSize = imageSize;
            layer.arc = arc;
        end
        
         function Z = predict(layer, X) % training
            % matlab中上一层输入的特征X为H*W*C*N，即h*w*c*bs
            X = dlarray(X);
            layer.numY = size(X,1);
            layer.numX = size(X,2);
            bs = size(X,4);
            layer.stride = max(layer.imageSize)./max(layer.numX,layer.numY);
            
            Z = X;
            % 2020.4.23 实际写下面2行的话，待整个yolov3网络predict时候报错！！！故移到外面写
%             Z = X.reshape(layer.numY,layer.numX,layer.nAnchors,(5+layer.nClasses),bs); 
%             Z = permute(Z,[5,3,1,2,4]);% 输出特征图，Z=bs*na*h*w*(5+nc),
         end
    
    end
end
