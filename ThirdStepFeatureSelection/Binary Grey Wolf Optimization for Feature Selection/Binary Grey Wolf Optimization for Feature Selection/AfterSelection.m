%%feature selectiondan sonra yapýlacaklar

TestFeat=testFeaturesGoogleNet(:,Sf);
classifier=fitcecoc(sFeat,trainingLabels)
YPred = predict(classifier,TestFeat);
accuracy = mean(YPred == testLabels)

TestFeat=testFeaturesGoogleNet(:,Sf);
classifier=fitcecoc(sFeat(:,1:100),trainingLabels)
YPred = predict(classifier,TestFeat(:,1:100));
accuracy100 = mean(YPred == testLabels)
% 
TestFeat=testFeaturesGoogleNet(:,Sf);
classifier=fitcecoc(sFeat(:,1:200),trainingLabels)
YPred = predict(classifier,TestFeat(:,1:200));
accuracy200 = mean(YPred == testLabels)

Dogruluk=[Nf  accuracy100 accuracy200 accuracy  ]
