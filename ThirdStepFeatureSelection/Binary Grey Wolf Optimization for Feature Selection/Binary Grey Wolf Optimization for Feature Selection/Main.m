%-------------------------------------------------------------------------%
%  Binary Grey Wolf Optimization (BGWO) source codes demo version         %
%                                                                         %
%  Programmer: Jingwei Too                                                %
%                                                                         %
%  E-Mail: jamesjames868@gmail.com                                        %
%-------------------------------------------------------------------------%

clc, clear, close 

% load 'C:\Users\dl4cv\Documents\MATLAB\ThirdStepFeatureSelection\EFeats\EAlexNetFeat.mat'
% load 'C:\Users\dl4cv\Documents\MATLAB\ThirdStepFeatureSelection\EFeats\EVGG19Feat.mat'
% load 'C:\Users\dl4cv\Documents\MATLAB\ThirdStepFeatureSelection\EFeats\EResNetFeat.mat'
load 'C:\Users\dl4cv\Documents\MATLAB\ThirdStepFeatureSelection\EFeats\EGoogleNetFeat.mat'

% feat=AlexNetFeatures; label=Labels; 
% feat=ResNetFeatures; label=Labels; 
% feat=VGG19Features; label=Labels; 
feat=GoogleNetFeatures; label=Labels; 
% feat:  feature vector (instances x features)
% label: labelling
% N:     Number of wolves
% T:     Maximum number of iterations
% *Note: k-value of KNN & k-fold setting can be modified in jFitnessFunction.m
%---Output-----------------------------------------------------------------
% sFeat: Selected features (instances x features)
% Sf:    Selected feature index
% Nf:    Number of selected features
% curve: Convergence curve
%--------------------------------------------------------------------------

%% (Method 1) BGWO1
close all; N=20; T=100;
[sFeat,Sf,Nf,curve]=jBGWO1(feat,label,N,T); 

% Plot convergence curve
% figure(); plot(1:T,curve); xlabel('Number of Iterations');
% ylabel('Fitness Value'); title('BGWO1'); grid on;

%% (Method 2) BGWO2
% close all; N=10; T=100; 
% [sFeat,Sf,Nf,curve]=jBGWO2(feat,label,N,T); 
% 
% % Plot convergence curve
% figure(); plot(1:T,curve); xlabel('Number of Iterations');
% ylabel('Fitness Value'); title('BGWO2'); grid on;




