clear variables;    % This is similar to 'clear all' but more efficient.
close all;
clc;

% Load image
myImgBarcode = imread('barcode_cropped.jpg');
figure(1);
subplot(2, 4, 1);
imshow(myImgBarcode);
title('Original Image');

barcodeGray = rgb2gray(myImgBarcode);
subplot(2, 4, 2);
imshow(barcodeGray);
title('Grascale');

barcodeEdgeSobel = edge(barcodeGray,"sobel");
subplot(2, 4, 3);
imshow(barcodeEdgeSobel);
title('Sobel');

barcodeEdgeCanny = edge(barcodeGray,"canny");
subplot(2, 4, 4);
imshow(barcodeEdgeCanny);
title('Canny');



% Load image
myImgChocolate = imread('chocolate_original.jpg');
subplot(2, 4, 5);
imshow(myImgChocolate);
title('Original Image');

chocolateGray = rgb2gray(myImgChocolate);
subplot(2, 4, 6);
imshow(chocolateGray);
title('Grascale');

chocolateEdgeSobel = edge(chocolateGray,"sobel");
subplot(2, 4, 7);
imshow(chocolateEdgeSobel);
title('Sobel');

chocolateEdgeCanny = edge(chocolateGray,"canny");
subplot(2, 4, 8);
imshow(chocolateEdgeCanny);
title('Canny');

