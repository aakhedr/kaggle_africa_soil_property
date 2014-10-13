function [error_train, error_val] = learningCurve_NN(X, y, Xval, yval, lambda, ...
	initial_nn_params, input_layer_size, hidden_layer_size, num_labels)

	m = size(X, 1); 			% number of training examples
    error_train = zeros(m, 1); error_val = zeros(size(Xval, 1), 1);

	for i = 1:m
		nn_params = trainNN(X(1:i, :), y(1:i, :), initial_nn_params, ... 
			input_layer_size, hidden_layer_size, num_labels, lambda);

		error_train(i) = nnCostFunction(X(1:i, :), y(1:i, :), nn_params, 0);
		error_val(i) = nnCostFunction(Xval, yval, nn_params, 0);
	end
end