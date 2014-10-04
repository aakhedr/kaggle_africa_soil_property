function [lambda_vec, error_train, error_val] = validationCurve(X, y, Xval, yval)
	lambda_vec = [0 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3];

	error_train = zeros(length(lambda_vec), 1);
	error_val = zeros(length(lambda_vec), 1);

	for i = 1:length(lambda_vec)
		theta = trainLinearReg(X, y, lambda_vec(i));

		error_train(i) = linRegCostFunc(X, y, theta, 0);
		error_val(i) = linRegCostFunc(Xval, yval, theta, 0);
	end
end