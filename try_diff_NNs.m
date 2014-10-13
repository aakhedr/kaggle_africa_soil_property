clear; close all; clc;

data = csvread('training.csv', 1, 1);

input_layer_size  = 3593;			% Number of features
hidden_layer_size = 150;
num_labels = 1;						% one of prediction values

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
initial_Theta3 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:); initial_Theta3(:)];

X = data(1:695, 1:3593); m = size(X, 1); X = [ones(m, 1) X]; lambda = .001;

Xval = data(696:926, 1:3593);   Xval = [ones(size(Xval, 1), 1) Xval];
Xtest = data(927:end, 1:3593);  Xtest = [ones(size(Xtest, 1), 1) Xtest];

for i = 1:5
	if i == 1
		y = data(1:695, 3595);		% Ca label only

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

Ca_SSE = sum((Ca - data(696:926, 3595)).^2);
P_SSE = sum((P - data(696:926, 3596)).^2);
pH_SSE = sum((pH - data(696:926, 3597)).^2);
SOC_SSE = sum((SOC - data(696:926, 3598)).^2);
SAND_SSE = sum((SAND - data(696:926, 3599)).^2);

fprintf('Ca squared error: %f\n', Ca_SSE);
fprintf('P squared error: %f\n', P_SSE);
fprintf('pH squared error: %f\n', pH_SSE);
fprintf('SOC squared error: %f\n', SOC_SSE);
fprintf('SAND squared error: %f\n', SAND_SSE);
fprintf('Total squared error: %f\n', (Ca_SSE + P_SSE + pH_SSE + SOC_SSE + SAND_SSE));
