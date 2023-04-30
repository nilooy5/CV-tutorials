% Week 13 example code for training a simple CNN classifier on the
% CUB_200_2011_Subset20classes dataset. Instead of using the full images,
% we only use the bounding box area defined in bounding_boxes.txt. This
% code gives an example of fivefold cross-validation in a way that ensures
% that each fold is used exactly once as validation fold and exactly once
% as test fold, while being used 3x as training fold.
%
% Authors: Roland Goecke and James Ireland. 
% Date created: 03/05/2022
% Date last updated: 23/04/23

close all;
clear variables;
existing_GUIs = findall(0);
if length(existing_GUIs) > 1
    delete(existing_GUIs);
end
clc;

%% Read the image data from the relevant text files. 
%  *** Adjust the file path as required. ***
folder = "..\week9\CUB_200_2011_Subset20classes\";
imgFolder = folder + "images/";
imgTxtFolder = folder + "images.txt";

% Load in all images from the dataset folder into one datastore
allImageDS = imageDatastore(imgFolder, 'IncludeSubfolders', true, ...
                            'LabelSource','foldernames');

%% Split dataset into five folds (=partitions) for fivefold cross-validation.
% Split dataset into 5 x 20% - Note, splitEachLabel splits the datastore
% into N+1 new datastores, so by specifying 0.2 four times, we will end up
% with five 20% partitions.
[fold1DS, fold2DS, fold3DS, fold4DS, fold5DS] = ...
    splitEachLabel(allImageDS, 0.2, 0.2, 0.2, 0.2);

% Set target size for common width and height after cropping
targetSize = [224, 224];

% Number of folds is five in this experiment
numFolds = 5;

%% Create a simple CNN
layers = [
    imageInputLayer([224 224 3])    % This needs to match the image size chosen above
    
    convolution2dLayer(3, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    
    fullyConnectedLayer(20)
    softmaxLayer
    classificationLayer];

%% Check if we have a GPU available and clear any old data from it
if (gpuDeviceCount() > 0)
    device = gpuDevice(1);
    reset(device);  % Clear previous values that might still be on the GPU
end

%% Train the simple CNN model for each fold
accuracy_overall = 0.0;
for i = 1:numFolds
    [cdsTraining, cdsValidation, cdsTest, trainingImageDS, ...
        validationImageDS, testImageDS] = ...
        getFoldsFor5FoldCrossVal(i, fold1DS, fold2DS, fold3DS, fold4DS, ...
                                 fold5DS, folder, imgTxtFolder, targetSize);

    % Set the training options
    options = trainingOptions('sgdm', ...
        'InitialLearnRate', 0.001, ...
        'MiniBatchSize', 20, ...
        'MaxEpochs', 10, ...
        'Verbose', false, ...
        'Shuffle', 'every-epoch', ...
        'VerboseFrequency', 1, ...
        'ValidationData', cdsValidation, ...
        'Plots','training-progress');

    simpleCNN = trainNetwork(cdsTraining, layers, options);

    YPred = classify(simpleCNN, cdsTest);
    YTest = testImageDS.Labels;
    
    accuracy = sum(YPred == YTest)/numel(YTest); % Output on command line
    disp("Accuracy for Run "+ string(i)+" is: " + accuracy);

    % Show confusion matrix in figure
    [m, order] = confusionmat(YTest, YPred);
    figure(i);
    cm = confusionchart(m, order, ...
        'ColumnSummary','column-normalized', ...
        'RowSummary','row-normalized');
    title("Overall Accuracy for Run "+ string(i)+" : "+ ...
        string(round(accuracy*100, 1)) +"%");

    accuracy_overall = accuracy_overall+accuracy;
end

disp("Average accuracy of five folds is "+ string(accuracy_overall/numFolds))
