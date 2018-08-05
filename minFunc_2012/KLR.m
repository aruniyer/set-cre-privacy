function[uRBF, preds, probs] = KLR(X, y, Xt, yt, scale, maxFunEvals)
% nVars = size(X, 2);
% nClasses = size(unique(y), 1);
nInstances = size(X, 1);
% X = [X ones(nInstances, 1)];
% Xt = [Xt ones(size(Xt, 1), 1)];
lambda = 1e-2;
options = [];
options.display = 'none';
options.maxFunEvals = maxFunEvals;

% First fit a regular linear model
% funObj = @(w)LogisticLoss(w,X,y);
% fprintf('Training linear logistic regression model...\n');
% wLinear = minFunc(@penalizedL2,zeros(nVars,1),options,funObj,lambda);

% Now fit the same model with the kernel representation
% K = kernelLinear(X,X);
% funObj = @(u)LogisticLoss(u,K,y);
% fprintf('Training kernel(linear) logistic regression model...\n');
% uLinear = minFunc(@penalizedKernelL2,zeros(nInstances,1),options,K,funObj,lambda);

% Now try a degree-2 polynomial kernel expansion
% polyOrder = 2;
% Kpoly = kernelPoly(X,X,polyOrder);
% funObj = @(u)LogisticLoss(u,Kpoly,y);
% fprintf('Training kernel(poly) logistic regression model...\n');
% uPoly = minFunc(@penalizedKernelL2,zeros(nInstances,1),options,Kpoly,funObj,lambda);

% Squared exponential radial basis function kernel expansion
rbfScale = scale;
Krbf = kernelRBF(X,X,rbfScale);
Ktrbf = kernelRBF(Xt,X,rbfScale);
funObj = @(u)LogisticLoss(u,Krbf,y);
fprintf('Training kernel(rbf) logistic regression model...\n');
uRBF = minFunc(@penalizedKernelL2,zeros(nInstances,1),options,Krbf,funObj,lambda);

% Check that wLinear and uLinear represent the same model:
% fprintf('Parameters estimated from linear and kernel(linear) model:\n');
% [wLinear X'*uLinear]

% trainErr_linear = sum(y ~= sign(X*wLinear))/length(y)
% trainErr_poly = sum(y ~= sign(Kpoly*uPoly))/length(y)
trainErr_rbf = sum(y ~= sign(Krbf*uRBF))/length(y)
testErr_rbf = sum(yt ~= sign(Ktrbf*uRBF))/length(y)
f = Ktrbf*uRBF;
preds = sign(f);
probs = exp(f)./(ones(size(f)) + exp(f));