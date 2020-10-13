%%Modelleri covide göre eðitme

clear all
imds = imageDatastore('..\..\dataset\orginial', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%%Her labeldeki resimlerin sayýsýný bul
tbl = countEachLabel(imds)

% Determine the smallest amount of images in a category
minSetCount = min(tbl{:,2}); 

% Limit the number of images to reduce the time it takes
% run this example.
maxNumImages =364;
minSetCount = min(maxNumImages,minSetCount);

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)
%%
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');
%%
numTrainImages = numel(imdsTrain.Labels);

%%
net = resnet50;

%%analyzeNetwork(net)

%%
inputSize = net.Layers(1).InputSize

lgraph = layerGraph(net);

% lgraph = removeLayers(lgraph, {'loss3-classifier','prob','output'});
lgraph = removeLayers(lgraph, {'fc1000_softmax','ClassificationLayer_fc1000'});
numClasses = numel(categories(imdsTrain.Labels));
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'fc1000','fc');

layers = lgraph.Layers;
connections = lgraph.Connections;

layers(1:110) = freezeWeights(layers(1:110));
lgraph = createLgraphUsingConnections(layers,connections);
% pixelRange = [-30 30];
% imageAugmenter = imageDataAugmenter( ...
%     'RandXReflection',true, ...
%     'RandXTranslation',pixelRange, ...
%     'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'ColorPreprocessing', 'gray2rgb');

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation,'ColorPreprocessing', 'gray2rgb');


 
 options = trainingOptions('sgdm', ...
    'MaxEpochs',50, ...
    'MiniBatchSize',64, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'InitialLearnRate',1e-5, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',3, ...
    'Verbose',1, ...
    'ValidationData',augimdsValidation);

%%

[netResNet,traininfo]  = trainNetwork(augimdsTrain,lgraph,options);

[YPred,scores] = classify(netResNet,augimdsValidation);
%%


%%
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)

save NetResNet.mat netResNet traininfo
