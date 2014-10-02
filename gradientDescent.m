function [theta, J_history] = gradientDescent(X, y, theta, alpha, lambda, ... 
	num_iters)

	J_history = zeros(num_iters, 1);
	m = size(X, 1);

	for iter = 1:num_iters
		predictions = X * theta;
		delta = (predictions - y)' * X;
		theta = theta - alpha * 1/ m * delta';

		[J_history(iter, :), grad] = linRegCostFunc(X, y, theta, lambda);
	end
end
