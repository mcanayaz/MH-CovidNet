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
maxNumImages = 2000;
minSetCount = min(maxNumImages,minSetCount);

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)
%%
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.7,'randomized');

%%
net = vgg19;

%%analyzeNetwork(net)

%%
inputSize = net.Layers(1).InputSize

layersTransfer = net.Layers(1:end-2);
%%
numClasses = numel(categories(imdsTrain.Labels))

layers = [
    layersTransfer
    reluLayer
    dropoutLayer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',10,'BiasLearnRateFactor',10)
    softmaxLayer
    classificationLayer];

layers(1:10) = freezeWeights(layers(1:10));
%%
% pixelRange = [-30 30];
% imageAugmenter = imageDataAugmenter( ...
%     'RandXReflection',true, ...
%     'RandXTranslation',pixelRange, ...
%     'RandYTranslation',pixelRange);
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

[netVGG19,traininfo] = trainNetwork(augimdsTrain,layers,opts);
[YPred,scores] = classify(netVGG19,augimdsValidation);
%%


%%
YValidation = imdsValidation.Labels;
accuracy = mean(YPred == YValidation)
save NetVGG19.mat netVGG19 traininfo
