%%Modelleri covide göre eðitme

clear all
imds = imageDatastore('..\..\dataset\en2', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');

%%Her labeldeki resimlerin sayýsýný bul
tbl = countEachLabel(imds)

% Determine the smallest amount of images in a category
minSetCount = min(tbl{:,2}); 

% Limit the number of images to reduce the time it takes
% run this example.
maxNumImages = 364;
minSetCount = min(maxNumImages,minSetCount);

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)
%%
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

%%
net = alexnet;

%%analyzeNetwork(net)

%%
inputSize = net.Layers(1).InputSize

layersTransfer = net.Layers(1:end-2);
%%
numClasses = numel(categories(imdsTrain.Labels))

layers = [
    layersTransfer
    
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer
    classificationLayer];
%%
% pixelRange = [-30 30];
% imageAugmenter = imageDataAugmenter( ...
%     'RandXReflection',true, ...
%     'RandXTranslation',pixelRange, ...
%     'RandYTranslation',pixelRange);
layers(1:10) = freezeWeights(layers(1:10));
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain,...
   'ColorPreprocessing', 'gray2rgb');
%%

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation,'ColorPreprocessing', 'gray2rgb');

%%
opts = trainingOptions('sgdm', ...
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

[netAlexNet,traininfo] = trainNetwork(augimdsTrain,layers,opts);
[YPred,scores] = classify(netAlexNet,augimdsValidation);
%

%%
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)

save NetAlexNet.mat netAlexNet traininfo
