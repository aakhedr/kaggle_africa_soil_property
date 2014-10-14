function [error_train, error_val] = learningCurve(X, y, Xval, yval, lambda)
	m = size(X, 1); 			% number of training examples
	error_train = zeros(m, 1); error_val = zeros(size(Xval, 1), 1);

	for i = 1:m
		theta = trainLinearReg(X(1:i, :), y(1:i, :), lambda);

		error_train(i) = linRegCostFunc(X(1:i, :), y(1:i, :), theta, 0);
		error_val(i) = linRegCostFunc(Xval, yval, theta, 0);
	end
end