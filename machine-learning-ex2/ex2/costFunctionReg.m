function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



temp_theta =  theta;
temp_theta(1)= 0;
J = (-1/m) *sum(y.*log(sigmoid(X *  theta))+ (1-y).*(log(1-sigmoid(X * theta)))) + (lambda/(2*m))* sum(temp_theta.^2);

%error = (sigmoid(X * theta))-y;
%temp0 = (1/m) * sum(error .*X(:,1));
%temp1 = (1/m) * sum(error .*X(:,2)) + (lambda/m)*theta(2);
%temp2 = (1/m) * sum(error .*X(:,3)) + (lambda/m)*theta(3);

%grad = [temp0;temp1;temp2];

error = (sigmoid(X * theta))-y;
theta(1) = 0;
grad = (1 / m) * (X' * error) + (lambda/m)*theta;


% =============================================================

end
