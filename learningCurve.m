function [error_train, error_val] = learningCurve(X, y, Xval, yval, lambda)
	m = size(X, 1); 			% number of training examples

	for i = 1:m
		theta = trainLinearReg(X, y, lambda);

		error_train(i) = linRegCostFunc(X, y, theta, 0);
		error_val(i) = linRegCostFunc(Xval, yval, theta, 0);
	end
end