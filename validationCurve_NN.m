function [lambda_vec, error_train, error_val] = validationCurve_NN(X, y, Xval, ...
	yval, initial_nn_params, input_layer_size, hidden_layer_size, num_labels)

	lambda_vec = [0 0.0001 0.0003 0.001 0.003 0.01 0.03 0.1 0.3];

	error_train = zeros(length(lambda_vec), 1);
	error_val = zeros(length(lambda_vec), 1);

	for i = 1:length(lambda_vec)
		nn_params = trainNN(initial_nn_params, input_layer_size, ...
			hidden_layer_size, num_labels, X, y, lambda_vec(i));

		error_train(i) = nnCostFunction(nn_params, input_layer_size, ...
			hidden_layer_size, num_labels, X, y, 0);
		error_val(i) = nnCostFunction(nn_params, input_layer_size, ...
			hidden_layer_size, num_labels, Xval, yval, 0);
	end
end