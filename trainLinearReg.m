function [theta] = trainLinearReg(X, y, lambda)

	% Initialize Theta
	initial_theta = zeros(size(X, 2), size(y, 2));

	% Create "short hand" for the cost function to be minimized
	costFunction = @(t) linRegCostFunc(X, y, t, lambda);

	% Now, costFunction is a function that takes in only one argument
	options = optimset('MaxIter', 200, 'GradObj', 'on');

	% Minimize using fmincg
	theta = fmincg(costFunction, initial_theta, options);
end
