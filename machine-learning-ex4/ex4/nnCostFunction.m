function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

grad_accum1 = zeros(size(Theta1));
grad_accum2 = zeros(size(Theta2));


% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
X = [ones(m,1) X];
zee2 = (X * Theta1');
a2 = sigmoid(zee2);

a2 = [ones(m,1) a2];
zee3 = (a2 * Theta2');
pred = sigmoid(zee3);

theta1_reg = Theta1;
theta1_reg(:,1) = 0;

theta2_reg = Theta2;
theta2_reg(:,1) = 0;

y = repmat([1:num_labels], m, 1) == repmat(y, 1, num_labels);

 J  = (-1/m)*sum(sum(y.*log(pred) +  (1-y).*log(1- pred))) +  (lambda/(2*m)*(sum(sum(theta1_reg.^2)))) + (lambda/(2*m)*(sum(sum(theta2_reg.^2))));


% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.


for a = 1:m

     a1t =  X(a,:);
     zee2t = zee2(a,:);
     a2t = a2(a,:);
     zee3t = zee3(a,:);
     predt = pred(a,:);
     yt = y(a,:);
    
    dzee3 = (predt - yt)';
    dzee2 = Theta2' * dzee3 .* sigmoidGradient([1; zee2t']);
    
    grad_accum1 = grad_accum1 + (dzee2(2:end) * a1t);
    
    grad_accum2 = grad_accum2 + (dzee3 * a2t);
     
end



%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Theta1_grad = 1/m * (grad_accum1) + (lambda/m)*theta1_reg;
Theta2_grad = 1/m * (grad_accum2) + (lambda/m)*theta2_reg;


















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
