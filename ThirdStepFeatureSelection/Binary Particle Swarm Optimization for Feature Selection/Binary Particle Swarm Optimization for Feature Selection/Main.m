%-------------------------------------------------------------------------%
%  Binary Particle Swarm Optimization (BPSO) source codes demo version    %
%                                                                         %
%  Programmer: Jingwei Too                                                %
%                                                                         %
%  E-Mail: jamesjames868@gmail.com                                        %
%-------------------------------------------------------------------------%

%clc, clear, close 
% Benchmark data set contains 351 instances and 34 features (binary class)
%load covid3.mat; % Matlab also provides this dataset (load Ionosphere.mat)
% Call features & labels
% load 'C:\Users\dl4cv\Documents\MATLAB\ThirdStepFeatureSelection\EFeats\EAlexNetFeat.mat'
% load 'C:\Users\dl4cv\Documents\MATLAB\ThirdStepFeatureSelection\EFeats\EVGG19Feat.mat'
load 'C:\Users\dl4cv\Documents\MATLAB\ThirdStepFeatureSelection\EFeats\EResNetFeat.mat'
% load 'C:\Users\dl4cv\Documents\MATLAB\ThirdStepFeatureSelection\EFeats\EGoogleNetFeat.mat'
% feat=trainingFeaturesVGG19; label=trainingLabels; 
% feat=AlexNetFeatures; label=Labels; 
feat=ResNetFeatures; label=Labels; 
% feat=VGG19Features; label=Labels; 
% feat=GoogleNetFeatures; label=Labels; 
%---Input------------------------------------------------------------------
% feat:  feature vector (instances x features)
% label: labelling
% N:     Number of particles
% T:     Maximum number of iterations
% c1:    Cognitive factor
% c2:    Social factor
% Vmax:  Maximum velocity
% Wmax:  Maximum bound on inertia weight
% Wmin:  Minimum bound on inertia weight
% *Note: k-value of KNN & k-fold setting can be modified in jFitnessFunction.m
%---Output-----------------------------------------------------------------
% sFeat: Selected features (instances x features)
% Sf:    Selected feature index
% Nf:    Number of selected features
% curve: Convergence curve
%--------------------------------------------------------------------------

%% (Method 1) BPSO
N=20; T=100; c1=2; c2=2; Vmax=6; Wmax=0.9; Wmin=0.4; 
[sFeat,Sf,Nf,curve]=jBPSO(feat,label,N,T,c1,c2,Wmax,Wmin,Vmax); 

% Plot convergence curve
% figure(); plot(1:T,curve); xlabel('Number of Iterations');
% ylabel('Fitness Value'); title('BPSO'); grid on;






