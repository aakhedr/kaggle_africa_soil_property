clear; close all; clc;

data = csvread('training.csv', 1, 1);

input_layer_size  = 3594;			% Number of features
hidden_layer_size = 250;			% The more the better
num_labels = 1;						% Ca prediction

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

X = data(:, 1:3594); [m, n] = size(X); X = [ones(m, 1) X]; lambda = 0.03;

testData = csvread('sorted_test.csv', 1, 1); [k, l] = size(testData);
Xtest = [ones(k, 1) testData];

for i = 1:5
	if i == 1
		y = data(:, 3595);		% Ca label only

		nn_params = trainNN(initial_nn_params, input_layer_size, hidden_layer_size, ...
			num_labels, X, y, lambda);

		Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
		                 hidden_layer_size, (input_layer_size + 1));

		Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
		                 num_labels, (hidden_layer_size + 1));

		h1 = Xtest * Theta1';
		Ca = [ones(size(h1, 1), 1) h1] * Theta2';

	elseif i == 2
		y = data(:, 3596);		% P label only

		nn_params = trainNN(initial_nn_params, input_layer_size, hidden_layer_size, ...
			num_labels, X, y, lambda);

		Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
		                 hidden_layer_size, (input_layer_size + 1));

		Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
		                 num_labels, (hidden_layer_size + 1));

		h1 = Xtest * Theta1';
		P = [ones(size(h1, 1), 1) h1] * Theta2';

	elseif i == 3
		y = data(:, 3597);		% pH label only

		nn_params = trainNN(initial_nn_params, input_layer_size, hidden_layer_size, ...
			num_labels, X, y, lambda);

		Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
		                 hidden_layer_size, (input_layer_size + 1));

		Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
		                 num_labels, (hidden_layer_size + 1));

		h1 = Xtest * Theta1';
		pH = [ones(size(h1, 1), 1) h1] * Theta2';

	elseif i == 4
		y = data(:, 3598);		% SOC label only

		nn_params = trainNN(initial_nn_params, input_layer_size, hidden_layer_size, ...
			num_labels, X, y, lambda);

		Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
		                 hidden_layer_size, (input_layer_size + 1));

		Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
		                 num_labels, (hidden_layer_size + 1));

		h1 = Xtest * Theta1';
		SOC = [ones(size(h1, 1), 1) h1] * Theta2';

	elseif i == 5
		y = data(:, 3599);		% SAND label only

		nn_params = trainNN(initial_nn_params, input_layer_size, hidden_layer_size, ...
			num_labels, X, y, lambda);

		Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
		                 hidden_layer_size, (input_layer_size + 1));

		Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
		                 num_labels, (hidden_layer_size + 1));

		h1 = Xtest * Theta1';
		SAND = [ones(size(h1, 1), 1) h1] * Theta2';
	end
end

csvwrite('submission3.csv', [Ca P pH SOC SAND]);

