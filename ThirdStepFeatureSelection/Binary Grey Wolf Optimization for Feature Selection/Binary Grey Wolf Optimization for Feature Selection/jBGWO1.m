%-------------------------------------------------------------------------%
%  Binary Grey Wolf Optimization (BGWO) source codes demo version         %
%                                                                         %
%  Programmer: Jingwei Too                                                %
%                                                                         %
%  E-Mail: jamesjames868@gmail.com                                        %
%-------------------------------------------------------------------------%

function [sFeat,Sf,Nf,curve]=jBGWO1(feat,label,N,T)
%---Inputs-----------------------------------------------------------------
% feat:  features
% label: labelling
% N:     Number of wolves
% T:     Maximum number of iterations
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
% Initial Population
X=zeros(N,D); fit=zeros(1,N);
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
% Sort fitness  
[~,idx]=sort(fit);  
% Update alpha, beta & delta wolves
Xalpha=X(idx(1),:); Xbeta=X(idx(2),:); Xdelta=X(idx(3),:);
Falpha=fit(idx(1)); Fbeta=fit(idx(2)); Fdelta=fit(idx(3));
% Pre
curve=inf; t=1; 
% figure(1); clf; axis([1 100 0 0.5]); xlabel('Number of Iterations');
% ylabel('Fitness Value'); title('Convergence Curve'); grid on;
%---Iterations start-------------------------------------------------------
while t <= T
	% Coefficient decreases linearly from 2 to 0 Eq(17)
    display(['Iteration:',num2str(t)])
  a=2-2*(t/T); 
  for i=1:N
    for d=1:D
      % Parameter C Eq(16)
      C1=2*rand(); C2=2*rand(); C3=2*rand();
      % Compute Dalpha, Dbeta & Ddelta Eq(22-24)
      Dalpha=abs(C1*Xalpha(d)-X(i,d)); Dbeta=abs(C2*Xbeta(d)-X(i,d));
      Ddelta=abs(C3*Xdelta(d)-X(i,d));
      % Parameter A Eq(15)
      A1=2*a*rand()-a; 
      % Compute Bstep 
      Bstep1=jBstepBGWO(A1*Dalpha); Bstep2=jBstepBGWO(A1*Dbeta); 
      Bstep3=jBstepBGWO(A1*Ddelta);
      % Wolf update 
      X1=jBGWOupdate(Xalpha(d),Bstep1); X2=jBGWOupdate(Xbeta(d),Bstep2);
      X3=jBGWOupdate(Xdelta(d),Bstep3);
      % Crossover update wolf Eq(25)
      X(i,d)=jCrossoverBGWO(X1,X2,X3);
    end
  end
  for i=1:N
    % Fitness 
    fit(i)=fun(feat,label,X(i,:));
    % Update alpha, beta & delta 
    if fit(i) < Falpha
      Falpha=fit(i); Xalpha=X(i,:);
    end
    if fit(i) < Fbeta && fit(i) > Falpha
      Fbeta=fit(i); Xbeta=X(i,:);
    end
    if fit(i) < Fdelta && fit(i) > Falpha && fit(i) > Fbeta
      Fdelta=fit(i); Xdelta=X(i,:);
    end
  end
  curve(t)=Falpha; 
  % Plot convergence curve
%   pause(0.000000001); hold on;
%   CG=plot(t,Falpha,'Color','r','Marker','.'); set(CG,'MarkerSize',5);
  t=t+1;
end
% Select features based on selected index 
Pos=1:D; Sf=Pos(Xalpha==1); Nf=length(Sf); sFeat=feat(:,Sf); 
end

%---Call Functions---------------------------------------------------------
function Bstep=jBstepBGWO(AD)
% Cstep Eq(28,31,34)
Cstep=1/(1+exp(-10*(AD-0.5))); 
% Bstep Eq(27,30,33)
if Cstep >= rand() 
	Bstep=1; 
else
	Bstep=0;
end
end

function Y=jBGWOupdate(X,Bstep) 
% Position update Eq(26,29,32)
if (X+Bstep) >= 1
	Y=1;
else
  Y=0;
end
end

function Y=jCrossoverBGWO(X1,X2,X3)
% Crossover Eq(35)
r=rand();
if r < 1/3
	Y=X1;
elseif r < 2/3 && r >=1/3
	Y=X2;
else
	Y=X3;
end
end

