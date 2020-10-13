%-------------------------------------------------------------------------%
%  Binary Particle Swarm Optimization (BPSO) source codes demo version    %
%                                                                         %
%  Programmer: Jingwei Too                                                %
%                                                                         %
%  E-Mail: jamesjames868@gmail.com                                        %
%-------------------------------------------------------------------------%                                                                         %

function [sFeat,Sf,Nf,curve]=jBPSO(feat,label,N,T,c1,c2,Wmax,Wmin,Vmax)
%---Inputs-----------------------------------------------------------------
% feat:  features
% label: labelling
% N:     Number of particles
% T:     Maximum number of iterations
% c1:    Cognitive factor
% c2:    Social factor
% Vmax:  Maximum velocity
% Wmax:  Maximum bound on inertia weight
% Wmin:  Minimum bound on inertia weight
%---Outputs----------------------------------------------------------------
% sFeat: Selected features
% Sf:    Selected feature index
% Nf:    Number of selected features
% curve: Convergence curve
%--------------------------------------------------------------------------

% Objective function
fun=@jFitnessFunction; 
% Number of dimensions
D=size(feat,2); 
% Initial Population: Position (X) & Velocity (V)
X=zeros(N,D); V=zeros(N,D); fit=zeros(1,N);
for i=1:N
  for d=1:D
    if rand() > 0.5
      X(i,d)=1;
    end
  end
end
% Fitness 
for i=1:N
  fit(i)=fun(feat,label,X(i,:)); 
end
% PBest & GBest
[fitG,idx]=min(fit); Xgb=X(idx,:); Xpb=X; fitP=fit; 
% Pre
curve=inf; t=1; 
% figure(1); clf; axis([1 100 0 0.5]); xlabel('Number of Iterations');
% ylabel('Fitness Value'); title('Convergence Curve'); grid on;
%---Iterations start-------------------------------------------------------
while t <= T
	% Inertia weight linearly decreases from 0.9 to 0.4 Eq(6)
    display (['Iteration: ',num2str(t)]);
  w=Wmax-(Wmax-Wmin)*(t/T);
  for i=1:N
      
    for d=1:D
      % Two random numbers in [0,1]
      r1=rand(); r2=rand();
      % Velocity update Eq(1)
      VB=V(i,d)*w+c1*r1*(Xpb(i,d)-X(i,d))+c2*r2*(Xgb(d)-X(i,d)); 
      % Limit velocity from overflying 
      VB(VB > Vmax)=Vmax; VB(VB < -Vmax)=-Vmax; V(i,d)=VB; 
      % Sigmoid function Eq(2)
      TF=1/(1+exp(-V(i,d)));
      % Position update Eq(3)
      if TF > rand()
        X(i,d)=1;
      else
        X(i,d)=0;
      end
    end
    % Fitness
    fit(i)=fun(feat,label,X(i,:));
    % Pbest update Eq(4)
    if fit(i) < fitP(i)
      Xpb(i,:)=X(i,:); fitP(i)=fit(i);
    end
    % Gbest update Eq(5)
    if fitP(i) < fitG
      Xgb=Xpb(i,:); fitG=fitP(i);
    end
    
  end
  curve(t)=fitG; 
  % Plot convergence curve
%   pause(0.000000001); hold on;
%   CG=plot(t,fitG,'Color','r','Marker','.'); set(CG,'MarkerSize',5);
  t=t+1;
end
% Select features 
Pos=1:D; Sf=Pos(Xgb==1); Nf=length(Sf); sFeat=feat(:,Sf); 
end



