function [nn_params] = trainNN(initial_nn_params, input_layer_size, ...
	hidden_layer_size, num_labels, X, y, lambda)

	fprintf('\nTraining Neural Network... \n')

	options = optimset('MaxIter', 50);

	% Create "short hand" for the cost function to be minimized
	costFunction = @(p) nnCostFunction(p, ...
	                                   input_layer_size, ...
	                                   hidden_layer_size, ...
	                                   num_labels, X, y, lambda);

	[nn_params] = fmincg(costFunction, initial_nn_params, options);

end