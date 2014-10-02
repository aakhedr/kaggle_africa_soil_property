function [J, grad] = linRegCostFunc(X, y, theta, lambda)
	m = size(X, 1);
	predictions = X * theta;

	% Sum of squared errors (SSE)
	SSE = sum((predictions - y).^2);

	% Regularization term
	costReg = lambda/(2 * m) * sum(theta(2:end,:).^2);

	% Cost
	J = 1/ (2 * m) * SSE + costReg; 

						% Gradient
	% without regularization
	grad_0 = 1/ m  * (X(:,1)' * (predictions - y));
	% with regularization
	gradReg = lambda/ m * theta(2:end,:);
	grad_rest = 1/ m * (X(:,2:end)' * (predictions - y)) + gradReg;
	
	grad = [grad_0; grad_rest];
end