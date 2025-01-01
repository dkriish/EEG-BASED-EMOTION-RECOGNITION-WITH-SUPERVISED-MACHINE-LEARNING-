% Load the DREAMER dataset
load('C:\Users\mages\OneDrive\Documents\krish Dissertation\DREAMER.mat');

% Initialize variables for features and labels
X = []; % Feature matrix
y_valence = []; % Valence score vector
y_arousal = []; % Arousal score vector
y_dominance = []; % Dominance score vector

% EEG Parameters
fs = DREAMER.EEG_SamplingRate; % Sampling rate
[b, a] = butter(4, [0.5 50] / (fs / 2), 'bandpass'); % Bandpass filter coefficients

% Process each participant and each stimulus
for subj = 1:DREAMER.noOfSubjects
    for clip = 1:DREAMER.noOfVideoSequences
        % Get EEG data for the current clip
        eegData = DREAMER.Data{subj}.EEG.stimuli{clip}; % EEG stimuli data
        filteredEEG = filtfilt(b, a, eegData); % Apply bandpass filter

        % Extract features for each EEG channel
        features = [];
        for ch = 1:size(filteredEEG, 2)
            signal = filteredEEG(:, ch);
            psd = pwelch(signal); % Power spectral density
            features = [features, ...
                mean(signal), var(signal), skewness(signal), kurtosis(signal), ...
                bandpower(psd, fs, [8 12]), bandpower(psd, fs, [13 30])]; % Alpha & Beta power
        end

        % Append features to the feature matrix
        X = [X; features];

        % Get emotion scores: Valence, Arousal, and Dominance
        valenceScore = DREAMER.Data{subj}.ScoreValence(clip);
        arousalScore = DREAMER.Data{subj}.ScoreArousal(clip);
        dominanceScore = DREAMER.Data{subj}.ScoreDominance(clip);

        % Append the scores to the label vectors
        y_valence = [y_valence; valenceScore];
        y_arousal = [y_arousal; arousalScore];
        y_dominance = [y_dominance; dominanceScore];
    end
end

% Split data into training and testing sets (80-20 split)
cv = cvpartition(y_valence, 'HoldOut', 0.2);
X_train = X(training(cv), :);
y_train_valence = y_valence(training(cv), :);
y_train_arousal = y_arousal(training(cv), :);
y_train_dominance = y_dominance(training(cv), :);
X_test = X(test(cv), :);
y_test_valence = y_valence(test(cv), :);
y_test_arousal = y_arousal(test(cv), :);
y_test_dominance = y_dominance(test(cv), :);

% Dimensionality Reduction (PCA) - Reduce to 81 features
[coeff, score_train, ~] = pca(X_train); % PCA on training data
X_train_reduced = score_train(:, 1:81); % Take first 81 components

score_test = (X_test - mean(X_train)) * coeff; % Apply PCA transformation to test data
X_test_reduced = score_test(:, 1:81); % Take first 81 components

% Reshape data for LSTM: [num_features, num_time_steps]
X_train_lstm = X_train_reduced'; % Transpose for [features x time_steps]
X_test_lstm = X_test_reduced'; % Transpose for [features x time_steps]

% Convert to cell arrays
X_train_lstm_cell = num2cell(X_train_lstm, [1]); % Each cell is a [features x time_steps] matrix
X_test_lstm_cell = num2cell(X_test_lstm, [1]);   % Each cell is a [features x time_steps] matrix

% Ensure labels are column vectors for consistency
y_train_arousal = y_train_arousal(:);
y_test_arousal = y_test_arousal(:);




%% 1. Valence Prediction: Random Forest Regression
rfModel = fitrensemble(X_train, y_train_valence, ...
    'Method', 'Bag', 'NumLearningCycles', 100, 'Learners', 'Tree');
valencePredictions = predict(rfModel, X_test);

% Evaluate the model performance using RMSE
valenceRMSE = sqrt(mean((valencePredictions - y_test_valence).^2));
disp(['Valence RMSE (Random Forest): ', num2str(valenceRMSE)]);

% Visualize Actual vs Predicted Valence
figure;
scatter(y_test_valence, valencePredictions);
title('Valence: Actual vs Predicted');
xlabel('Actual Valence');
ylabel('Predicted Valence');
grid on;

%% 2. LSTM Model for Arousal Prediction
layers = [
    sequenceInputLayer(81) % 81 features
    lstmLayer(100, 'OutputMode', 'last') % LSTM with 100 units, output last time step
    fullyConnectedLayer(1) % Fully connected to output single value
    regressionLayer]; % Regression layer for continuous output

options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 64, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

% Train the LSTM network
lstmModel = trainNetwork(X_train_lstm_cell, y_train_arousal, layers, options);

% Predict on test data
arousalPredictions = predict(lstmModel, X_test_lstm_cell);

% Evaluate the predictions
arousalRMSE = sqrt(mean((arousalPredictions - y_test_arousal).^2));
disp(['Arousal RMSE (LSTM): ', num2str(arousalRMSE)]);

% Visualize Actual vs Predicted Arousal
figure;
scatter(y_test_arousal, arousalPredictions);
title('Arousal: Actual vs Predicted');
xlabel('Actual Arousal');
ylabel('Predicted Arousal');
grid on;


%% 3. Dominance Prediction: SVM Regression
svmModel = fitrsvm(X_train, y_train_dominance, ...
    'KernelFunction', 'rbf', 'Standardize', true);
dominancePredictions = predict(svmModel, X_test);

% Evaluate the model performance using RMSE
dominanceRMSE = sqrt(mean((dominancePredictions - y_test_dominance).^2));
disp(['Dominance RMSE (SVM Regression): ', num2str(dominanceRMSE)]);

% Visualize Actual vs Predicted Dominance
figure;
scatter(y_test_dominance, dominancePredictions);
title('Dominance: Actual vs Predicted');
xlabel('Actual Dominance');
ylabel('Predicted Dominance');
grid on;


%% Calculate the correlation between actual and predicted values for each emotion score
corr_valence = corr(y_test_valence, valencePredictions);
corr_arousal = corr(y_test_arousal, arousalPredictions);
corr_dominance = corr(y_test_dominance, dominancePredictions);

% Display the correlations in the command window
disp(['Correlation for Valence: ', num2str(corr_valence)]);
disp(['Correlation for Arousal: ', num2str(corr_arousal)]);
disp(['Correlation for Dominance: ', num2str(corr_dominance)]);

% Create a correlation matrix for all predictions and actual values
correlation_matrix = [
    corr(y_test_valence, valencePredictions), corr(y_test_valence, arousalPredictions), corr(y_test_valence, dominancePredictions);
    corr(y_test_arousal, valencePredictions), corr(y_test_arousal, arousalPredictions), corr(y_test_arousal, dominancePredictions);
    corr(y_test_dominance, valencePredictions), corr(y_test_dominance, arousalPredictions), corr(y_test_dominance, dominancePredictions)
];

% Plot the heatmap for the correlation matrix
figure;
heatmap({'Valence', 'Arousal', 'Dominance'}, {'Valence', 'Arousal', 'Dominance'}, correlation_matrix, ...
    'ColorbarVisible', 'on', 'Colormap', jet, 'Title', 'Correlation Heatmap for Actual vs Predicted');


%% Comparartive analysis
% Example 1: ANOVA Test (One-Way Analysis of Variance)
% for the comparison of the emotion scores (e.g., valence, arousal, dominance) based on different categories in the dataset.
% Let's say the data is grouped by a categorical variable, for example, a 'Group' variable in your dataset.

% Assuming 'Group' is a categorical variable that segments the data
% If no specific grouping variable exists, this section assumes using valence as an example.

% Example: Performing one-way ANOVA on the emotion scores (e.g., valence)
% Ensure that the 'Group' is a categorical variable that classifies the observations (e.g., 'Happy', 'Sad', etc.)
Group = categorical(randi([1, 3], length(y_valence), 1)); % Simulated grouping (3 groups)
% Perform ANOVA test on valence scores based on 'Group'
[p_anova, tbl_anova, stats_anova] = anova1(y_valence, Group);
disp('ANOVA Results for Valence:');
disp(tbl_anova);
disp(['p-value: ', num2str(p_anova)]);

% Perform post-hoc test if ANOVA is significant (optional)
if p_anova < 0.05
    disp('Performing Tukey''s Honest Significant Difference Test:');
    [c, m, h, gnames] = multcompare(stats_anova);
    disp(c);
end

% Example 2: Wilcoxon Signed-Rank Test (for Paired Samples)
% Assuming you want to compare two related samples (e.g., before and after treatment or two emotion scores)

% Wilcoxon test between Valence and Arousal (paired comparison)
% Ensure the two variables have the same length
[~, p_wilcoxon] = ranksum(y_valence, y_arousal);
disp('Wilcoxon Signed-Rank Test:');
disp(['p-value: ', num2str(p_wilcoxon)]);

% If you want to perform the Wilcoxon test on the same data for paired samples:
% Use the `signrank` function to test the differences between two related samples
% Example: Perform Wilcoxon test on 'valence' and 'arousal' for paired samples
[p_wilcoxon_signed, h_wilcoxon_signed] = signrank(y_valence, y_arousal);
disp('Wilcoxon Signed-Rank Test for Paired Samples:');
disp(['p-value: ', num2str(p_wilcoxon_signed)]);
disp(['Test Result (h): ', num2str(h_wilcoxon_signed)]);

% If the p-value from Wilcoxon test is less than 0.05, we reject the null hypothesis (i.e., significant difference exists)
if p_wilcoxon_signed < 0.05
    disp('There is a significant difference between Valence and Arousal.');
else
    disp('There is no significant difference between Valence and Arousal.');
end




% Visualizing the distribution of emotion scores (Valence, Arousal, Dominance)
figure;
subplot(3,1,1);
histogram(y_valence, 20); % Histogram of Valence scores
title('Distribution of Valence Scores');
xlabel('Valence');
ylabel('Frequency');

subplot(3,1,2);
histogram(y_arousal, 20); % Histogram of Arousal scores
title('Distribution of Arousal Scores');
xlabel('Arousal');
ylabel('Frequency');

subplot(3,1,3);
histogram(y_dominance, 20); % Histogram of Dominance scores
title('Distribution of Dominance Scores');
xlabel('Dominance');
ylabel('Frequency');

% Visualize the relationship between emotion scores using pairwise scatter plots
figure;
subplot(1,3,1);
scatter(y_valence, y_arousal);
title('Valence vs Arousal');
xlabel('Valence');
ylabel('Arousal');
grid on;

subplot(1,3,2);
scatter(y_valence, y_dominance);
title('Valence vs Dominance');
xlabel('Valence');
ylabel('Dominance');
grid on;

subplot(1,3,3);
scatter(y_arousal, y_dominance);
title('Arousal vs Dominance');
xlabel('Arousal');
ylabel('Dominance');
grid on;

% Visualize feature distribution for the first few features in the dataset (e.g., mean, variance)
% Select first few features from the feature matrix X
X_subset = X(:, 1:5); % First 5 features for visualization
figure;
subplot(2,3,1);
boxplot(X_subset(:,1)); % Boxplot of Mean
title('Mean Feature Distribution');

subplot(2,3,2);
boxplot(X_subset(:,2)); % Boxplot of Variance
title('Variance Feature Distribution');

subplot(2,3,3);
boxplot(X_subset(:,3)); % Boxplot of Skewness
title('Skewness Feature Distribution');

subplot(2,3,4);
boxplot(X_subset(:,4)); % Boxplot of Kurtosis
title('Kurtosis Feature Distribution');

subplot(2,3,5);
boxplot(X_subset(:,5)); % Boxplot of Alpha Band Power
title('Alpha Band Power Feature Distribution');

% Visualizing the correlation between the first few features
corr_matrix = corr(X_subset); % Compute correlation matrix of selected features

% Visualize the correlation matrix using a heatmap
figure;
heatmap({'Mean', 'Variance', 'Skewness', 'Kurtosis', 'Alpha Power'}, ...
    {'Mean', 'Variance', 'Skewness', 'Kurtosis', 'Alpha Power'}, ...
    corr_matrix, 'ColorbarVisible', 'on', 'Colormap', jet, ...
    'Title', 'Feature Correlation Heatmap');

% Pairwise scatter plot of the first few features
figure;
subplot(2,3,1);
scatter(X_subset(:,1), X_subset(:,2));
title('Mean vs Variance');
xlabel('Mean');
ylabel('Variance');
grid on;

subplot(2,3,2);
scatter(X_subset(:,1), X_subset(:,3));
title('Mean vs Skewness');
xlabel('Mean');
ylabel('Skewness');
grid on;

subplot(2,3,3);
scatter(X_subset(:,1), X_subset(:,4));
title('Mean vs Kurtosis');
xlabel('Mean');
ylabel('Kurtosis');
grid on;

subplot(2,3,4);
scatter(X_subset(:,1), X_subset(:,5));
title('Mean vs Alpha Power');
xlabel('Mean');
ylabel('Alpha Power');
grid on;

subplot(2,3,5);
scatter(X_subset(:,2), X_subset(:,3));
title('Variance vs Skewness');
xlabel('Variance');
ylabel('Skewness');
grid on;

subplot(2,3,6);
scatter(X_subset(:,2), X_subset(:,4));
title('Variance vs Kurtosis');
xlabel('Variance');
ylabel('Kurtosis');
grid on;
