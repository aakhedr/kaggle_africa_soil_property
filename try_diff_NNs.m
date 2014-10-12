clear; close all; clc;

data = csvread('training.csv', 1, 1);

input_layer_size  = 3593;			% Number of features
hidden_layer_size = 250;			% The more the better
num_labels = 1;						% Ca prediction

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

X = data(1:695, 1:3593); m = size(X, 1); X = [ones(m, 1) X]; lambda = 0.03;

Xval = data(696:926, 1:3593);   Xval = [ones(size(Xval, 1), 1) Xval];
Xtest = data(927:end, 1:3593);  Xtest = [ones(size(Xval, 1), 1) Xtest];

for i = 1:5
	if i == 1
		y = data(1:695, 3595);		% Ca label only
        yval = data(696:926, 3595);
        ytest = data(927:end, 3595);

		nn_params = trainNN(initial_nn_params, input_layer_size, hidden_layer_size, ...
			num_labels, X, y, lambda);

		Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
		                 hidden_layer_size, (input_layer_size + 1));

		Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
		                 num_labels, (hidden_layer_size + 1));

		h1 = Xval * Theta1';
		Ca = [ones(size(h1, 1), 1) h1] * Theta2';

	elseif i == 2
		y = data(1:695, 3596);		% P label only
        yval = data(696:926, 3596);
        ytest = data(927:end, 3596);

		nn_params = trainNN(initial_nn_params, input_layer_size, hidden_layer_size, ...
			num_labels, X, y, lambda);

		Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
		                 hidden_layer_size, (input_layer_size + 1));

		Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
		                 num_labels, (hidden_layer_size + 1));

		h1 = Xval * Theta1';
		P = [ones(size(h1, 1), 1) h1] * Theta2';

	elseif i == 3
		y = data(1:695, 3597);		% pH label only
        yval = data(696:926, 3597);
        ytest = data(927:end, 3597);

		nn_params = trainNN(initial_nn_params, input_layer_size, hidden_layer_size, ...
			num_labels, X, y, lambda);

		Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
		                 hidden_layer_size, (input_layer_size + 1));

		Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
		                 num_labels, (hidden_layer_size + 1));

		h1 = Xval * Theta1';
		pH = [ones(size(h1, 1), 1) h1] * Theta2';

	elseif i == 4
		y = data(1:695, 3598);		% SOC label only
        yval = data(696:926, 3598);
        ytest = data(927:end, 3598);

		nn_params = trainNN(initial_nn_params, input_layer_size, hidden_layer_size, ...
			num_labels, X, y, lambda);

		Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
		                 hidden_layer_size, (input_layer_size + 1));

		Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
		                 num_labels, (hidden_layer_size + 1));

		h1 = Xval * Theta1';
		SOC = [ones(size(h1, 1), 1) h1] * Theta2';

	elseif i == 5
		y = data(1:695, 3599);		% SAND label only
        yval = data(696:926, 3599);
        ytest = data(927:end, 3599);

		nn_params = trainNN(initial_nn_params, input_layer_size, hidden_layer_size, ...
			num_labels, X, y, lambda);

		Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
		                 hidden_layer_size, (input_layer_size + 1));

		Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
		                 num_labels, (hidden_layer_size + 1));

		h1 = Xval * Theta1';
		SAND = [ones(size(h1, 1), 1) h1] * Theta2';
	end
end
Ca_SSE = sum((Ca - yval).^2);
P_SSE = sum((P - yval).^2);
pH_SSE = sum((pH - yval).^2);
SOC_SSE = sum((SOC - yval).^2);
SAND_SSE = sum((SAND - yval).^2);

fprintf('Ca squared error: %f\n', Ca_SSE);
fprintf('P squared error: %f\n', P_SSE);
fprintf('pH squared error: %f\n', pH_SSE);
fprintf('SOC squared error: %f\n', SOC_SSE);
fprintf('SAND squared error: %f\n', SAND_SSE);
fprintf('Total squared error: %f\n', (Ca_SSE + P_SSE + pH_SSE + SOC_SSE + SAND_SSE));
