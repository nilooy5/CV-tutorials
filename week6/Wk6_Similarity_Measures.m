clear variables;    % This is similar to 'clear all' but more efficient.
close all;
clc;

% Load image
robocupImg = imread('images\robocup_image1.jpeg');

figure(1);
subplot(2, 2, 1);
imshow(robocupImg);
title('Original Image');

robocupImgGrey = rgb2gray(robocupImg);
subplot(2, 2, 2);
imshow(robocupImgGrey);
title('Greyscale Image');

templateBallGrayImg = robocupImgGrey(405:450,97:140); % IF THIS IS NOT AN EVEN NUMBER YOU'RE IN TROUBLE
subplot(2, 2, 3);
imshow(templateBallGrayImg);
title('Greyscale Image');


% Compute SAD values
[SAD_values, tmpHeightHalf, tmpWidthHalf, imgSize] = compute_SAD(robocupImgGrey, templateBallGrayImg);
maxSAD = max(max(SAD_values)); % Need this for scaling of the values
[minSAD_Val, minSAD_Col] = min(min(SAD_values(tmpHeightHalf : imgSize(1)-tmpHeightHalf, tmpWidthHalf : imgSize(2)-tmpWidthHalf)));
[minSAD_Val, minSAD_Row] = min(SAD_values(tmpHeightHalf : imgSize(1)-tmpHeightHalf, minSAD_Col+tmpWidthHalf-1));
% Top-left corner coordinates of best fitting location
disp([minSAD_Row minSAD_Col minSAD_Val]);