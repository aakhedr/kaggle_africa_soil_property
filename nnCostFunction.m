function [J, grad] = nnCostFunction(nn_params, input_layer_size, ...
	hidden_layer_size, num_labels, X, y, lambda)

	Theta1 = reshape(nn_params(1:hidden_layer_size * ...
		(input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

	Theta2 = reshape(nn_params((1 + (hidden_layer_size * ...
		(input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

	%==========================================================================
	% Forward propagation
	%==========================================================================
	m = size(X, 1);

	a1 = X; 										% Input unit
	z2 = a1 * Theta1'; a2 = [ones(m, 1) z2];		% Hidden unit
	z3 = a2 * Theta2';								% Output unit

	% Regularization term
	costReg = lambda/ (2 * m) * ... 
		(sum(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))));

	% Sum of squared erros
	SSE = sum((z3 - y).^2);
	J = 1/ (2 * m) * SSE + costReg; 
	%==========================================================================
	% Back propagation
	%==========================================================================


end