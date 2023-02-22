clear variables;    % This is similar to 'clear all' but more efficient.
close all;
clc;

% Load image
myImg = imread('barcode_cropped.jpg');
figure(1);
subplot(4, 5, 1);
imshow(myImg);
title('Original Image');

[r,g,b] = imsplit(myImg);

subplot(4, 5, 2);
imshow(r);
title('Red channel');

subplot(4, 5, 3);
imshow(g);
title('Green channel');

subplot(4, 5, 4);
imshow(b);
title('Blue channel');

myImgGray = rgb2gray(myImg);
subplot(4,5,5);
imshow(myImgGray);
title('Gray');

subplot(4,5,7);
imhist(r);
title('hist for R');

subplot(4,5,8);
imhist(g);
title('hist for G');

subplot(4,5,9);
imhist(b);
title('hist for Blue');

subplot(4,5,10);
imhist(myImgGray);
title('hist for GrayScale');

myImgRedHistEq = histeq(r);
myImgBlueHistEq = histeq(g);
myImgGreenHistEq = histeq(b);
myImgGrayHistEq = histeq(myImgGray);

subplot(4,5,12);
imshow(myImgRedHistEq);
title('Red (hist Eq)');

subplot(4,5,13);
imshow(myImgGreenHistEq);
title('Green (hist Eq)');

subplot(4,5,14);
imshow(myImgBlueHistEq);
title('Blue (hist Eq)');

subplot(4,5,15);
imshow(myImgGrayHistEq);
title('Gray (hist Eq)');


subplot(4,5,17);
imhist(myImgRedHistEq);
subplot(4,5,18);
imhist(myImgGreenHistEq);
subplot(4,5,19);
imhist(myImgBlueHistEq);
subplot(4,5,20);
imhist(myImgGrayHistEq);

myImgHistEq = zeros(size(myImg), 'uint8');
myImgHistEq(:,:,1) = myImgRedHistEq;
myImgHistEq(:,:,2) = myImgGreenHistEq;
myImgHistEq(:,:,3) = myImgBlueHistEq;
subplot(4,5,16);
imshow(myImgHistEq);
title('After hist eq')
