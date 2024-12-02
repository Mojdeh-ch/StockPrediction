% NARX Neural Network

% Load the dataset
data = readtable('b.csv');
head(data)

% Identify and convert character columns to numeric

characterColumns = {'tsi', 'pi', 'turnover', 'not', 'cd', 'tvmv', 'Inflation', 'cpi', 'fund', 'pe', 'finalpriceabid'};
for i = 1:numel(characterColumns)
    colName = characterColumns{i};
    if iscell(data.(colName)) || isstring(data.(colName))
        data.(colName) = str2double(data.(colName));
    end
end
%% 

% Define independent variables
independentVars = {'tsi', 'pi', 'turnover', 'not', 'cd', 'tvmv', 'Inflation', 'cpi', 'fund', 'pe'};
X = log(data{:, independentVars}); % Extract independent variables

% Define the dependent variable
y = log(data.finalpriceabid);

% Define the split points
train_end = 1773;
%val_end = 355;
test_end = 313;
Q = 2086;

% Ensure the total number of samples matches the sum of train, val, and test sizes
assert(train_end + test_end == Q, 'The sizes must sum up to the total number of samples');

% Assign the indices
train_indices = 1:train_end;
%val_indices = train_end + 1:train_end+val_end;
test_indices = train_end + 1:Q;

% Divide indices into training, validation, and test sets
[trainInd, testInd] = divideind(Q,train_indices,test_indices);
%% 
%   X - input time series.
%   y - feedback time series.
X = tonndata(X,false,false);
T = tonndata(y,false,false);

%% 
trainFcn = 'trainbfg';

% Create a Nonlinear Autoregressive Network with External Input
inputDelays = 1:8;
feedbackDelays = 1:10;

hiddenLayerSize = [10,6];
net = narxnet(inputDelays, feedbackDelays, hiddenLayerSize, 'open', trainFcn);
net.trainParam.epochs = 200;
%net.trainParam.mu = 0.001;        % Initial learning rate
%net.trainParam.mu_inc = 1.02;     % Learning rate increase factor

net.layers{1}.transferFcn = 'purelin';
net.layers{2}.transferFcn = 'purelin';
% Apply L2 regularization
net.performParam.regularization = 0.1;

net.divideFcn = 'divideind';
net.divideMode = 'time';
net.divideParam.trainInd = train_indices;
%net.divideParam.valInd = val_indices;
net.divideParam.testInd = test_indices;

[x,xi,ai,t] = preparets(net,X,{},T);

% Train the Network
[net,tr] = train(net,x,t,xi,ai);
% Calculate the predictions for training data
trainOutputs = net(x(:, tr.trainInd), xi, ai);

% Transform predictions and targets back to the original scale
trainTargets = exp(cell2mat(t(:, tr.trainInd)));
trainOutputs = exp(cell2mat(trainOutputs));

% Calculate RMSE
train_rmse = sqrt(mean((trainTargets - trainOutputs).^2));

% Calculate RMSLE
train_rmsle = sqrt(mean((log1p(trainTargets) - log1p(trainOutputs)).^2));

% Calculate MAE
train_mae = mean(abs(trainTargets - trainOutputs));

% Calculate MAPE
train_mape = mean(abs((trainTargets - trainOutputs) ./ trainTargets)) * 100;

disp('Training Data Metrics (Open Loop):');
disp(['RMSE: ', num2str(train_rmse)]);
disp(['RMSLE: ', num2str(train_rmsle)]);
disp(['MAE: ', num2str(train_mae)]);
disp(['MAPE: ', num2str(train_mape)]);
%% 
% Close the loop for prediction
netc = closeloop(net);
netc.name = [net.name ' - Closed Loop'];

% Initialize state for test prediction
[~,xi_test,ai_test] = preparets(netc, X(:,train_indices),{},T(:,train_indices));

% Predict on test data
testPredictions = netc(X(:, test_indices), xi_test, ai_test);

% Transform predictions and targets back to the original scale
testTargets = exp(cell2mat(T(:, test_indices)));
testPredictions = exp(cell2mat(testPredictions));

% Calculate RMSE
test_rmse = sqrt(mean((testTargets - testPredictions).^2));

% Calculate RMSLE
test_rmsle = sqrt(mean((log1p(testTargets) - log1p(testPredictions)).^2));

% Calculate MAE
test_mae = mean(abs(testTargets - testPredictions));

% Calculate MAPE
test_mape = mean(abs((testTargets - testPredictions) ./ testTargets)) * 100;

disp('Test Data Metrics (Closed Loop):');
disp(['RMSE: ', num2str(test_rmse)]);
disp(['RMSLE: ', num2str(test_rmsle)]);
disp(['MAE: ', num2str(test_mae)]);
disp(['MAPE: ', num2str(test_mape)])
%% 
% Assuming testPredictions is a matrix or array
filename = 'testPredictions_NARX.xlsx';

% Write to Excel file
writematrix(testPredictions, filename);

%% 
% Plot Actual vs Predicted Values
figure;
plot(testTargets, 'b', 'DisplayName', 'Actual');
hold on;
plot(testPredictions, 'r', 'DisplayName', 'Predicted');
xlabel('Time');
ylabel('Value');
title('Actual vs Predicted Values');
legend('show');
grid on;

%% 
% Closed Loop Network
netc = closeloop(net);
netc.name = [net.name ' - Closed Loop'];
view(netc)
[xc,xic,aic,tc] = preparets(netc,X,{},T);
yc = netc(xc,xic,aic);
closedLoopPerformance = perform(net,tc,yc);





