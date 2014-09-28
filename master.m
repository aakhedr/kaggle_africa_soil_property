% Initialization
clear, close all; clc

% read data into MATLAB
% Topsoil=1 & Subsoil=0
data = csvread('training.csv', 1, 1)

% train data = 60% of total (~= 695 rows)
train = data(1:695, :);
x = train(:, 1:3594); y = train(:, 3595:end);
[m, n] = size(x);