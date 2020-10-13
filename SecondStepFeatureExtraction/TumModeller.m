
%%Baþlangýç Resim Yükleme
% imageFolder='C:\Users\dl4cv\Documents\MATLAB\dataset\orginial'
% imds = imageDatastore(imageFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);


imageFolder='C:\Users\dl4cv\Documents\MATLAB\dataset\enhancement'
imds = imageDatastore(imageFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);



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

%% Split Data
% Load pretrained network

[trainingSet, testSet] = splitEachLabel(imds, 0.7, 'randomize');

%% ResNet
% Create augmentedImageDatastore from training and test sets to resize
% images in imds to the size required by the network.

%ResNet
% load('C:\Users\dl4cv\Documents\MATLAB\SecondStepFeatureExtraction\originalresults\OResNet.mat')
load('C:\Users\dl4cv\Documents\MATLAB\SecondStepFeatureExtraction\enhancementresults\EResNet.mat')
netResNet = netResNet;

imageSizeResNet = netResNet.Layers(1).InputSize;

augmentedTrainingSetResNet = augmentedImageDatastore(imageSizeResNet, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSetResNet = augmentedImageDatastore(imageSizeResNet, testSet, 'ColorPreprocessing', 'gray2rgb');


featureLayerResNet = 'fc1000';
trainingFeaturesResNet = activations(netResNet, augmentedTrainingSetResNet, featureLayerResNet, ...
    'MiniBatchSize', 32, 'OutputAs', 'rows');

testFeaturesResNet= activations(netResNet, augmentedTestSetResNet, featureLayerResNet, ...
    'MiniBatchSize', 32, 'OutputAs', 'rows');
%% AlexNet
%AlexNet
% load('C:\Users\dl4cv\Documents\MATLAB\SecondStepFeatureExtraction\originalresults\OAlexNet.mat')
load('C:\Users\dl4cv\Documents\MATLAB\SecondStepFeatureExtraction\enhancementresults\EAlexNet.mat')
netAlexNet=netAlexNet;
imageSizeAlexNet=netAlexNet.Layers(1).InputSize;
augmentedTrainingSetAlexNet = augmentedImageDatastore(imageSizeAlexNet, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSetAlexNet = augmentedImageDatastore(imageSizeAlexNet, testSet, 'ColorPreprocessing', 'gray2rgb');


featureLayerAlexNet = 'fc8';
trainingFeaturesAlexNet = activations(netAlexNet, augmentedTrainingSetAlexNet, featureLayerAlexNet, ...
    'MiniBatchSize', 32, 'OutputAs', 'rows');

% Extract test features using the CNN
testFeaturesAlexNet= activations(netAlexNet, augmentedTestSetAlexNet, featureLayerAlexNet, ...
    'MiniBatchSize', 32, 'OutputAs', 'rows');

%% VGG19
%VGG19
% load('C:\Users\dl4cv\Documents\MATLAB\SecondStepFeatureExtraction\originalresults\OVGG19.mat')
load('C:\Users\dl4cv\Documents\MATLAB\SecondStepFeatureExtraction\enhancementresults\EVGG19.mat')
netVGG19=netVGG19;
imageSizeVGG19=netVGG19.Layers(1).InputSize;
augmentedTrainingSetVGG19 = augmentedImageDatastore(imageSizeVGG19, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSetVGG19 = augmentedImageDatastore(imageSizeVGG19, testSet, 'ColorPreprocessing', 'gray2rgb');



featureLayerVGG19 = 'fc8';
trainingFeaturesVGG19 = activations(netVGG19, augmentedTrainingSetVGG19, featureLayerVGG19, ...
    'MiniBatchSize', 32, 'OutputAs', 'rows');

% Extract test features using the CNN
testFeaturesVGG19= activations(netVGG19, augmentedTestSetVGG19, featureLayerVGG19, ...
    'MiniBatchSize', 32, 'OutputAs', 'rows');

%% GoogleNet
% load('C:\Users\dl4cv\Documents\MATLAB\SecondStepFeatureExtraction\originalresults\OGoogleNet.mat')
load('C:\Users\dl4cv\Documents\MATLAB\SecondStepFeatureExtraction\enhancementresults\EGoogleNet.mat')
netGoogleNet=netGoogleNet;
imageSizeGoogleNet=netGoogleNet.Layers(1).InputSize;
augmentedTrainingSetGoogleNet = augmentedImageDatastore(imageSizeGoogleNet, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSetGoogleNet = augmentedImageDatastore(imageSizeGoogleNet, testSet, 'ColorPreprocessing', 'gray2rgb');



featureLayerGoogleNet = 'loss3-classifier' ;
trainingFeaturesGoogleNet = activations(netGoogleNet, augmentedTrainingSetGoogleNet, featureLayerGoogleNet, ...
    'MiniBatchSize', 32, 'OutputAs', 'rows');

% Extract test features using the CNN
testFeaturesGoogleNet= activations(netGoogleNet, augmentedTestSetGoogleNet, featureLayerGoogleNet, ...
    'MiniBatchSize', 32, 'OutputAs', 'rows');


%% Feature Combine

% Get training labels from the trainingSet
trainingLabels = trainingSet.Labels;

testLabels = testSet.Labels;

% TrainingFeaturesCombine=[trainingFeaturesAlexNet trainingFeaturesVGG19 trainingFeaturesResNet];
% TestingFeaturesCombine=[testFeaturesAlexNet testFeaturesVGG19 testFeaturesResNet];
% 
% save covidFeaturesCombine.mat TrainingFeaturesCombine trainingLabels TestingFeaturesCombine testLabels 


%% Classifier for ResNet
clear classifierResNet

classifierResNet = fitcecoc(trainingFeaturesResNet , trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'rows');
 
 
predictedLabelsResNet = predict(classifierResNet, testFeaturesResNet, 'ObservationsIn', 'rows');
 

confMatResNet = confusionmat(testLabels, predictedLabelsResNet); 

sum(diag(confMatResNet))/sum(confMatResNet(:))

confusionchart(testLabels,predictedLabelsResNet);

figure;
cmResNet=confusionchart(testLabels,predictedLabelsResNet);
cmResNet.Title="ResNet";
cmR = confusionchart(testLabels,predictedLabelsResNet,'RowSummary','row-normalized','ColumnSummary','column-normalized');
cmR.Title="ResNet";


%% Classifier for AlexNet



clear classifierAlexNet

classifierAlexNet = fitcecoc(trainingFeaturesAlexNet , trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'rows');
 
 
predictedLabelsAlexNet = predict(classifierAlexNet, testFeaturesAlexNet, 'ObservationsIn', 'rows');
 

confMatAlexNet = confusionmat(testLabels, predictedLabelsAlexNet); 

sum(diag(confMatAlexNet))/sum(confMatAlexNet(:))

confusionchart(testLabels,predictedLabelsAlexNet);

figure;
cmALexNet=confusionchart(testLabels,predictedLabelsAlexNet);
cmALexNet.Title="AlexNet";
figure;
cmA = confusionchart(testLabels,predictedLabelsAlexNet,'RowSummary','row-normalized','ColumnSummary','column-normalized');
cmA.Title="AlexNet";

%% Classifier for VGG19
clear classifierVGG19

classifierVGG19 = fitcecoc(trainingFeaturesVGG19 , trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'rows');
 
 
predictedLabelsVGG19 = predict(classifierVGG19, testFeaturesVGG19, 'ObservationsIn', 'rows');
 

confMatVGG19 = confusionmat(testLabels, predictedLabelsVGG19); 

sum(diag(confMatVGG19))/sum(confMatVGG19(:))

figure;
cmVGG19=confusionchart(testLabels,predictedLabelsVGG19);
cmVGG19.Title="VGG19";
figure;
cmV = confusionchart(testLabels,predictedLabelsVGG19,'RowSummary','row-normalized','ColumnSummary','column-normalized');
cmV.Title="VGG19";


%% Classifier for GoogleNet
clear classifierGoogleNet

classifierGoogleNet = fitcecoc(trainingFeaturesGoogleNet , trainingLabels, ...
    'Learners', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'rows');
 
 
predictedLabelsGoogleNet = predict(classifierGoogleNet, testFeaturesGoogleNet, 'ObservationsIn', 'rows');
 

confMatGoogleNet = confusionmat(testLabels, predictedLabelsGoogleNet); 

sum(diag(confMatGoogleNet))/sum(confMatGoogleNet(:))

figure;
cmGoogleNet=confusionchart(testLabels,predictedLabelsGoogleNet);
cmGoogleNet.Title="GoogleNet";
figure;
cmV = confusionchart(testLabels,predictedLabelsGoogleNet,'RowSummary','row-normalized','ColumnSummary','column-normalized');
cmV.Title="GoogleNet";