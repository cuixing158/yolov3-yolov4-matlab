function [giouRatio,iouRatio] = getGIOU(bboxA,bboxB)
% 功能：获取bboxA和bboxB之间的GIOU或IOU，两两组合计算GIOU值
% 输入： 
%      bboxA, M*4大小矩阵，形式为[x,y,w,h]
%      bboxB, N*4大小矩阵，形式为[x,y,w,h]
% 输出：
%     giouRatio：M*N大小矩阵，每个元素的值表示所在的行列(i,j)分别表示来自bboxA,bboxB的第i,j个bbox的GIOU值
%     iouRatio：M*N大小矩阵，每个元素的值表示所在的行列(i,j)分别表示来自bboxA,bboxB的第i,j个bbox的IOU值
%
% 参考：1、https://arxiv.org/abs/1902.09630，CVPR2019,
%      《Generalized Intersection over Union: A Metric and A Loss for Bounding Box Regression 》
%       https://zhuanlan.zhihu.com/p/57992040
%
% cuixingxing150@gmail.com
% 2020.4.25
%
M = size(bboxA,1);
N = size(bboxB,1);
giouRatio = zeros(M,N,'like',bboxA);
iouRatio = zeros(M,N,'like',bboxA);

areasA = bboxA(:,3).*bboxA(:,4);
areasB = bboxB(:,3).*bboxB(:,4);
xyxyA = [bboxA(:,1:2),bboxA(:,1)+bboxA(:,3),bboxA(:,2)+bboxA(:,4)];
xyxyB = [bboxB(:,1:2),bboxB(:,1)+bboxB(:,3),bboxB(:,2)+bboxB(:,4)];

for i = 1:M
    for j = 1:N
        x1 = max(xyxyA(i,1),xyxyB(j,1));
        x2 = min(xyxyA(i,3),xyxyB(j,3));
        y1 = max(xyxyA(i,2),xyxyB(j,2));
        y2 = min(xyxyA(i,4),xyxyB(j,4));
        Intersection = max(0,(x2-x1)).*max(0,(y2-y1));
        Union = areasA(i)+areasB(j)-Intersection;
        iouRatio(i,j) = Intersection./Union;
        
        x1 = min(xyxyA(i,1),xyxyB(j,1));
        x2 = max(xyxyA(i,3),xyxyB(j,3));
        y1 = min(xyxyA(i,2),xyxyB(j,2));
        y2 = max(xyxyA(i,4),xyxyB(j,4));
        Convex = (x2-x1).*(y2-y1);
        giouRatio(i,j) = iouRatio(i,j) - (Convex-Union)./Convex;
    end
end







