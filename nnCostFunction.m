function [J, grad] = nnCostFunction(nn_params, input_layer_size, ...
	hidden_layer_size, num_labels, X, y, lambda)

	Theta1 = reshape(nn_params(1:hidden_layer_size * ...
		(input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

    Theta2 = reshape(nn_params(1 + (hidden_layer_size * (input_layer_size ... 
        + 1)): hidden_layer_size * (input_layer_size + 1) + (hidden_layer_size * ... 
        (hidden_layer_size + 1))), hidden_layer_size, hidden_layer_size + 1);
    
    Theta3 = reshape(nn_params(1 + (hidden_layer_size * (input_layer_size + 1)) ... 
        + (hidden_layer_size * (hidden_layer_size + 1)): end), num_labels, ... 
        hidden_layer_size + 1);

	%==========================================================================
	% Forward propagation
	%==========================================================================
	m = size(X, 1);

	a1 = X; 									% Input unit
	z2 = a1 * Theta1'; a2 = [ones(m, 1) z2];	% Hidden unit 1
    
	z3 = a2 * Theta2'; a3 = [ones(m, 1) z3];	% Hidden unit 2
    
    z4 = a3 * Theta3'; a4 = z4;                 % Output unit

	% Regularization term
	costReg = lambda/ (2 * m) * ... 
		(sum(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2))));

	% Sum of squared erros
	SSE = sum((z4 - y).^2);
	J = 1/ (2 * m) * SSE + costReg; 
	%==========================================================================
	% Back propagation
	%==========================================================================
	delta4 = a4 - y;
    delta3 = delta4 * Theta3(:, 2:end);
	delta2 = delta3 * Theta2(:, 2:end);

    DELTA3 = delta4' * a3; DELTA2 = delta3' * a2; DELTA1 = delta2' * a1;

	% Regularization term
	reg1 = lambda/ m * Theta1; 
    reg2 = lambda/ m * Theta2; 
    reg3 = lambda/ m * Theta3;
	
    reg1(:,1) = 0; reg2(:,1) = 0; reg3(:,1) = 0;

	% Gradient 
	Theta1_grad = 1/ m * DELTA1 + reg1; 
    Theta2_grad = 1/ m * DELTA2 + reg2;
    Theta3_grad = 1/ m * DELTA3 + reg3;

	%% Unroll gradient
	grad = [Theta1_grad(:); Theta2_grad(:); Theta3_grad(:)];
end