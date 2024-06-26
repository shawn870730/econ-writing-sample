% This is the coding sample of my writing sample
rGDP = readtable('rGDP.csv');          % 4 ~103
pi = readtable('inflation_Rates.csv'); % 2 ~101

ES = readtable('Employment_Service.csv');
Serv=readtable('Services.csv');
GDP_pc=readtable('GDP_per_capita.csv');
Export=readtable('Export.csv');

rGDP_matrix=rGDP{1:57,8:103}; % 4+4 ~103
log_rGDP_matrix=log(rGDP_matrix);
pi_matrix=pi{1:57,6:101};     % 2+4 ~101
pi_matrix_2=pi{1:57,2:101};   % for calculation

ES_matrix=ES{1:57,5:28};
Serv_matrix=Serv{1:57,5:29}; % NaN include
GDP_pc_matrix=GDP_pc{1:57,5:29};
Export_matrix=Export{1:57,5:29}; % NaN include


Epi_matrix = nan(size(pi_matrix, 1), size(pi_matrix, 2)); % lose data of 1998

for t = 5:size(pi_matrix_2, 2) 
    Epi_matrix(:, t-4) = mean(pi_matrix_2(:, (t-4):(t-1)), 2);
end

num_series = size(log_rGDP_matrix, 1);
% Smoothing parameter for yearly data
lambda = 1600;

% Initialize matrices for trend and cycle components
gdp_trend_components = zeros(size(log_rGDP_matrix));
gdp_cycle_components = zeros(size(log_rGDP_matrix));

% Apply the HP filter to each time series
for i = 1:num_series
    [trend, cycle] = hpfilter(log_rGDP_matrix(i, :)', lambda);
    gdp_trend_components(i, :) = trend';
    gdp_cycle_components(i, :) = cycle';
end

% Real_GDP-trend_components
output_gap_matrix=gdp_cycle_components*100;

% Initialize variables to store results
kappa_value = zeros(57, 1);
nwse = zeros(57, 1);

for i = 1:num_series
    X_1 = output_gap_matrix(i, :)';
    X_2 = Epi_matrix(i, :)';
    X=[X_1,X_2];
    Y = pi_matrix(i, :)';
    lm = fitlm(X, Y);
    [EstCoeffCov,se,coeff]=hac(lm, 'type', 'HC', 'weights', 'HC0','display','off');
    
    % Store the results
    kappa_value(i) = coeff(2); % Coefficient for X
    nwse(i)=se(2);
    
end

n=56;
k=3;
tstats = kappa_value ./ nwse;
pvalues = 2 * (1 - tcdf(abs(tstats), n - k));

num_positive_kappa = sum(kappa_value > 0);
num_significant_p_value = sum(pvalues <= 0.05);

row_averages_of_ES = mean(ES_matrix, 2);
row_averages_of_Serv = nanmean(Serv_matrix, 2);
row_averages_of_GDP_pc = mean(GDP_pc_matrix, 2);
row_averages_of_Export = nanmean(Export_matrix, 2);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

figure
sz=80;
scatter(row_averages_of_ES, kappa_value,sz,"filled");
xlabel('Employment in Services(%)');
ylabel('Kappa Value');
title('Scatter Plot of Kappa Value and Employment in Services');
grid on;

figure
scatter(row_averages_of_Serv, kappa_value,sz,"filled");
xlabel('Services (GDP%)');
ylabel('Kappa-Value');
title('Scatter Plot of Kappa Value and Services (GDP%)');
grid on;



significant_indices = pvalues <= 0.05;
filtered_kappa_value = kappa_value(significant_indices);

% Filter other variables if necessary
filtered_row_averages_of_ES = row_averages_of_ES(significant_indices);
filtered_row_averages_of_Serv = row_averages_of_Serv(significant_indices);
filtered_row_averages_of_GDP_pc = row_averages_of_GDP_pc(significant_indices);
% ... (filter other variables as needed)

% Plotting
figure
sz=80
scatter(filtered_row_averages_of_ES, filtered_kappa_value, sz, "filled");
xlabel('Employment in Services(%)');
ylabel('Kappa Value');
title('Scatter Plot of Kappa Value and Employment in Services');
grid on;

%figure
%bubblechart(row_averages_of_ES,kappa_value,row_averages_of_GDP_pc,'#7031BB');
%bubblelegend('GDP Per Capita','Location','eastoutside');
%xlabel('Employment in Services(%)');
%ylabel('Kappa-Value');
%title('Bubble Plot of Kappa Value and Employment Rate in Services');

figure
bubblechart(filtered_row_averages_of_ES,filtered_kappa_value,filtered_row_averages_of_GDP_pc,'#7031BB');
bubblelegend('GDP Per Capita','Location','eastoutside');
xlabel('Employment in Services(%)');
ylabel('Kappa-Value');
title('Bubble Plot of Kappa Value and Employment Rate in Services');

%figure
%scatter(filtered_row_averages_of_Serv, filtered_kappa_value, sz, "filled");
%xlabel('Services (GDP%)');
%ylabel('Kappa-Value');
%title('Scatter Plot of Kappa Value and Services (GDP%)');
%grid on;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Remove row 18 from filtered_kappa_value and filtered_row_averages_of_ES
filtered_kappa_value(18) = [];
filtered_row_averages_of_ES(18) = [];
filtered_row_averages_of_GDP_pc(18) = [];


lm = fitlm(filtered_row_averages_of_ES, filtered_kappa_value);
disp(lm);


% Predict kappa values using the fitted model
predicted_kappa_values = predict(lm, filtered_row_averages_of_ES);


figure;
scatter(filtered_row_averages_of_ES, filtered_kappa_value, 'filled');
hold on; 
plot(filtered_row_averages_of_ES, predicted_kappa_values, 'r-', 'LineWidth', 2);
xlabel('Employment in Services(%)');
ylabel('Kappa Value');
title('Scatter Plot of Kappa Value and Employment in Services');
grid on;
hold off; 



