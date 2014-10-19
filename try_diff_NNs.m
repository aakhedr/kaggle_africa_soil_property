clear; close all; clc;

data = csvread('training.csv', 1, 1);

input_layer_size  = 3593;			% Number of features
hidden_layer_size = 250;
num_labels = 1;						% one of prediction values

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, hidden_layer_size);
initial_Theta3 = randInitializeWeights(hidden_layer_size, num_labels);

initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:); initial_Theta3(:)];

X = data(1:810, 1:3593); m = size(X, 1); X = [ones(m, 1) X];

Xval = data(811:end, 1:3593);   Xval = [ones(size(Xval, 1), 1) Xval];

for i = 1:5
	if i == 1
		y = data(1:810, 3595);		% Ca label only
        lambda = .03;

		nn_params = trainNN(initial_nn_params, input_layer_size, hidden_layer_size, ...
			num_labels, X, y, lambda);

        Theta1 = reshape(nn_params(1:hidden_layer_size * ...
            (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

        Theta2 = reshape(nn_params(1 + (hidden_layer_size * (input_layer_size ... 
            + 1)): hidden_layer_size * (input_layer_size + 1) + (hidden_layer_size * ... 
            (hidden_layer_size + 1))), hidden_layer_size, hidden_layer_size + 1);

        Theta3 = reshape(nn_params(1 + (hidden_layer_size * (input_layer_size + 1)) ... 
            + (hidden_layer_size * (hidden_layer_size + 1)): end), num_labels, ... 
            hidden_layer_size + 1);

        h1 = Xval * Theta1';
        h2 = [ones(size(h1, 1), 1) h1] * Theta2';
		Ca = [ones(size(h2, 1), 1) h2] * Theta3';

    elseif i == 2
		y = data(1:810, 3596);		% P label only
        lambda = 10;

		nn_params = trainNN(initial_nn_params, input_layer_size, hidden_layer_size, ...
			num_labels, X, y, lambda);

        Theta1 = reshape(nn_params(1:hidden_layer_size * ...
            (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

        Theta2 = reshape(nn_params(1 + (hidden_layer_size * (input_layer_size ... 
            + 1)): hidden_layer_size * (input_layer_size + 1) + (hidden_layer_size * ... 
            (hidden_layer_size + 1))), hidden_layer_size, hidden_layer_size + 1);

        Theta3 = reshape(nn_params(1 + (hidden_layer_size * (input_layer_size + 1)) ... 
            + (hidden_layer_size * (hidden_layer_size + 1)): end), num_labels, ... 
            hidden_layer_size + 1);

		h1 = Xval * Theta1';
        h2 = [ones(size(h1, 1), 1) h1] * Theta2';
		P = [ones(size(h2, 1), 1) h2] * Theta3';

	elseif i == 3
		y = data(1:810, 3597);		% pH label only
        lambda = .1;

		nn_params = trainNN(initial_nn_params, input_layer_size, hidden_layer_size, ...
			num_labels, X, y, lambda);

        Theta1 = reshape(nn_params(1:hidden_layer_size * ...
            (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

        Theta2 = reshape(nn_params(1 + (hidden_layer_size * (input_layer_size ... 
            + 1)): hidden_layer_size * (input_layer_size + 1) + (hidden_layer_size * ... 
            (hidden_layer_size + 1))), hidden_layer_size, hidden_layer_size + 1);

        Theta3 = reshape(nn_params(1 + (hidden_layer_size * (input_layer_size + 1)) ... 
            + (hidden_layer_size * (hidden_layer_size + 1)): end), num_labels, ... 
            hidden_layer_size + 1);

		h1 = Xval * Theta1';
        h2 = [ones(size(h1, 1), 1) h1] * Theta2';
		pH = [ones(size(h2, 1), 1) h2] * Theta3';

	elseif i == 4
		y = data(1:810, 3598);		% SOC label only
        lambda = .0003;

		nn_params = trainNN(initial_nn_params, input_layer_size, hidden_layer_size, ...
			num_labels, X, y, lambda);

        Theta1 = reshape(nn_params(1:hidden_layer_size * ...
            (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

        Theta2 = reshape(nn_params(1 + (hidden_layer_size * (input_layer_size ... 
            + 1)): hidden_layer_size * (input_layer_size + 1) + (hidden_layer_size * ... 
            (hidden_layer_size + 1))), hidden_layer_size, hidden_layer_size + 1);

        Theta3 = reshape(nn_params(1 + (hidden_layer_size * (input_layer_size + 1)) ... 
            + (hidden_layer_size * (hidden_layer_size + 1)): end), num_labels, ... 
            hidden_layer_size + 1);

		h1 = Xval * Theta1';
        h2 = [ones(size(h1, 1), 1) h1] * Theta2';
		SOC = [ones(size(h2, 1), 1) h2] * Theta3';

	elseif i == 5
		y = data(1:810, 3599);		% SAND label only
        lambda = .003;

		nn_params = trainNN(initial_nn_params, input_layer_size, hidden_layer_size, ...
			num_labels, X, y, lambda);

        Theta1 = reshape(nn_params(1:hidden_layer_size * ...
            (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

        Theta2 = reshape(nn_params(1 + (hidden_layer_size * (input_layer_size ... 
            + 1)): hidden_layer_size * (input_layer_size + 1) + (hidden_layer_size * ... 
            (hidden_layer_size + 1))), hidden_layer_size, hidden_layer_size + 1);

        Theta3 = reshape(nn_params(1 + (hidden_layer_size * (input_layer_size + 1)) ... 
            + (hidden_layer_size * (hidden_layer_size + 1)): end), num_labels, ... 
            hidden_layer_size + 1);

		h1 = Xval * Theta1';
        h2 = [ones(size(h1, 1), 1) h1] * Theta2';
		SAND = [ones(size(h2, 1), 1) h2] * Theta3';
	end
end

Ca_SSE = sum((Ca - data(811:end, 3595)).^2);
P_SSE = sum((P - data(811:end, 3596)).^2);
pH_SSE = sum((pH - data(811:end, 3597)).^2);
SOC_SSE = sum((SOC - data(811:end, 3598)).^2);
SAND_SSE = sum((SAND - data(811:end, 3599)).^2);

fprintf('Ca squared error: %f\n', Ca_SSE);
fprintf('P squared error: %f\n', P_SSE);
fprintf('pH squared error: %f\n', pH_SSE);
fprintf('SOC squared error: %f\n', SOC_SSE);
fprintf('SAND squared error: %f\n', SAND_SSE);
fprintf('Total squared error: %f\n', (Ca_SSE + P_SSE + pH_SSE + SOC_SSE + SAND_SSE));
