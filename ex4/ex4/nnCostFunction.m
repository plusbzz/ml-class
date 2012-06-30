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

% First, recode Y!
y_rec = zeros(num_labels,m);

for i = 1:m,
  y_rec(y(i),i) = 1;
end

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%


% Compute J without regularization

A1 = [ones(m,1) X];
Z2 = A1*Theta1';
A2 = [ones(m,1) sigmoid(Z2)];
Z3 = A2*Theta2';
A3 = sigmoid(Z3);

J1 = sum(sum(y_rec'.* log(A3)));
J2 = sum(sum((1-y_rec').*log(1-A3)));

J = -(J1+J2)/m;

% Compute regularization parameter

% Layer 1
Theta1_trunc = Theta1(1:end,2:end); % skip first column
L1 = sum(sum(Theta1_trunc .* Theta1_trunc));

% Layer 2
Theta2_trunc = Theta2(1:end,2:end); % skip first column
L2 = sum(sum(Theta2_trunc .* Theta2_trunc));

Reg = (lambda * (L1+L2))/(2*m); % regularization
J = J + Reg;


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
%

% Layer 3
delta_3 = A3' - y_rec;
Del_2 = delta_3 * A2;
Theta2_grad = Del_2/m;

% Layer 2
delta_2 = ((delta_3'*Theta2_trunc) .* sigmoidGradient(Z2))';
Del_1 = delta_2 * A1;
Theta1_grad = Del_1/m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
