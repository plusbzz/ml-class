function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

	% Cost function
	hyp = sigmoid(X*theta);
	J_1 = (-y')*(log(hyp));
	J_0 = (1-y')*(log(1.-hyp));
	
	J_orig = (J_1-J_0)/m;

  theta_reg = theta(2:n);
  J_reg = (lambda*(theta_reg'*theta_reg))/(2*m);
	
	J = J_orig + J_reg;


	% Gradient
	reg_factor = (lambda/m).*theta;
	reg_factor(1) = 0;
	
	grad = ((hyp - y)'*X)/m + reg_factor';

% =============================================================

end
