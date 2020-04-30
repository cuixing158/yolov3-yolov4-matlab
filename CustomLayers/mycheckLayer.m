validInputSize = [3,3,18,1];
layer = yolov3Layer('yolov3', rand(3,2),1,1,[416,416],'default');
checkLayer(layer,validInputSize,'ObservationDimension' ,4)

%%
layer = mishLayer('mishACT');
validInputSize = [24 24 20];
checkLayer(layer,validInputSize,'ObservationDimension',4)


