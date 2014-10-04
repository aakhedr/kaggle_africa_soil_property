function [theta] = trainLinearReg(X, y, lambda)

	initial_theta = zeros(size(X, 2), size(y, 2));

	options = optimset('MaxIter', 200, 'GradObj', 'on');

	% Create "short hand" for the cost function to be minimized
	costFunction = @(t) linRegCostFunc(X, y, t, lambda);

	theta = fmincg(costFunction, initial_theta, options);
end
