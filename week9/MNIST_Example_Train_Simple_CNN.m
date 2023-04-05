% Example of Creating a Simple Convolutional Neural Network (CNN) model
% for the MNIST Handwritten Digits dataset
%
% Author: Roland Goecke
% Date created: 26/04/2022
% Date last changed: 03/04/2023

close all;
clear variables;
clc;

%% Load training data
% Define file names
trainImageFilename = 'train-images.idx3-ubyte';
trainLabelFilename = 'train-labels.idx1-ubyte';

% Process files and store training image and label information
[Xtrain,LabelTrain] = processMNISTdata_for_CNN(trainImageFilename,trainLabelFilename);

%% Load the test data
testImageFilename = 't10k-images.idx3-ubyte';
testLabelFilename = 't10k-labels.idx1-ubyte';
[Xtest,LabelTest] = processMNISTdata_for_CNN(testImageFilename,testLabelFilename);

%% Define CNN Model
% Deep learning models contain a mix of convolutional, batch normalisation,
% ReLU, and pooling layers... Here's an example.
layers = [
    imageInputLayer([28 28 1])
	
    convolution2dLayer(3,16,'Padding',1)
    batchNormalizationLayer
    reluLayer
	
    maxPooling2dLayer(2,'Stride',2)
	
    convolution2dLayer(3,32,'Padding',1)
    batchNormalizationLayer
    reluLayer
	
    maxPooling2dLayer(2,'Stride',2)
	
    convolution2dLayer(3,64,'Padding',1)
    batchNormalizationLayer
    reluLayer
	
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%% Check if we have a GPU available and clear any old data from it
if (gpuDeviceCount() > 0)
    disp('Found GPU:');
    disp(gpuDeviceTable);
    device = gpuDevice(1);
    reset(device);  % Clear previous values that might still be on the GPU
end

%% Train the CNN model (aka training the network)
% Define training parameters
miniBatchSize = 2000;
options = trainingOptions( 'sgdm',...
    'MaxEpochs', 25,...
    'InitialLearnRate',0.01,...
    'MiniBatchSize', miniBatchSize,...
    'Plots', 'training-progress');

% Train the CNN
model = trainNetwork(Xtrain, LabelTrain, layers, options);

%% Test the accuracy of the trained model on the test data
predLabelsTest = model.classify(Xtest);
accuracy = sum(predLabelsTest == LabelTest) / numel(LabelTest)

% Show confusion matrix in figure
[m, order] = confusionmat(LabelTest, predLabelsTest);
figure(1);
cm = confusionchart(m, order, ...
    'ColumnSummary','column-normalized', ...
    'RowSummary','row-normalized');

%% Test on a sample image
% Set i to the index of the image choosen
i = 2;
curimg = Xtest(:, :, i);
% We will copy and resize the image for demonstration purposes 
figure(2);
imshow(imresize(curimg,10), [0 255]);
% Grab the ground truth and compare to our prediction, if correct the
% predicted label should be the same as the annotated one. 
groundTruth = LabelTest(i);
predLabel = model.classify(curimg);
disp(append('Predicted label = ', string(predLabel),' vs. Ground Truth = ', string(groundTruth))) 
