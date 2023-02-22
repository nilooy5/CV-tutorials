% A MATLAB script for the homework of Week 2. Output should be viewed in
% a maximised figure window (full screen).
%
% Author: Roland Goecke
% Date created: 16/02/21
% Date last updated: 12/02/23

clear variables;    % This is similar to 'clear all' but more efficient.
close all;
clc;

% Load image
myImg = imread('chocolate_original.jpg');
figure(1);
subplot(4, 4, 1);
imshow(myImg);
title('Original Image');

% Separate the Red, Green and Blue channels
[myImgRed, myImgGreen, myImgBlue] = imsplit(myImg);

% Convert colour image to grayscale
myImgGray = rgb2gray(myImg);
subplot(4, 4, 2);
imshow(myImgGray);
title('Grayscale Image');

% Display the blue channel
subplot(4, 4, 3);
imshow(myImgBlue);
title('Blue channel');

% Show the image histogram of the blue channel
subplot(4, 4, 4);
imhist(myImgBlue);
title('Histogram Blue');

% Interrogating the image histogram, we see three peaks in the distribution
% of the pixel values in the blue channel image. Using Tools -> Data Tips
% in the output image, we can sample the pixel values at different
% locations and find that the first peak seems to mostly be pixels from the
% background (the table surface), that the second peak seems to mostly be
% pixels from the blue letters on the chocolate wrapper, and that the third
% peak corresponds mostly to the white chocolate wrapper.
% From this we deduct, that in order to extract the blue letters as much as
% possible, we want to keep the pixels whose values correspond to the
% second peak, while zeroing out all others. The lower threshold is around
% pixel value 33 and the upper threshold is around pixel value 88.
iLowerThreshold = 33;
iUpperThreshold = 88;

% Option 1 (slow!) - Comment out one of the two options - 
% Use two nested loops to check the pixels in all rows and 
% columns of the blue channel image. If the value is between 33 and 88,
% we see the binary mask image pixel at the same location to 1 (meaning
% this is a foreground pixel we want to keep), otherwise 0.
% [rows, cols] = size(myImgBlue);
% blueChannelBinaryMask = zeros(size(myImgBlue), 'logical');
% for iRow = 1:rows
%     for iCol = 1:cols
%         if (myImgBlue(iRow, iCol) >= iLowerThreshold) && ...
%                 (myImgBlue(iRow, iCol) <= iUpperThreshold)
%             blueChannelBinaryMask(iRow, iCol) = 1;
%         end
%     end
% end

% Option 2 (faster!) - Comment out one of the two options -
% Use a comparison and a bitwise AND.
blueChannelBinaryMask = (myImgBlue >= iLowerThreshold) & ...
                        (myImgBlue <= iUpperThreshold);

% Display the blue channel mask
subplot(4, 4, 5);
imshow(blueChannelBinaryMask);
title('Blue channel mask (binary)');

% Apply the binary mask to the blue channel image using an element-wise
% multiplication (this is what '.*'). Because the mask array is of type
% 'logical' (=binary), we need to type cast it to uint8 (8-bit unsigned 
% integer) to match the data type of 
myImgBlueMasked = myImgBlue .* cast(blueChannelBinaryMask, "uint8");
subplot(4, 4, 6);
imshow(myImgBlueMasked);
title('Blue with mask applied');

% Apply the binary mask also to the greyscale image. As we can see in the
% output, part of the table surface is still considered foreground.
myImgGrayMasked = myImgGray .* cast(blueChannelBinaryMask, "uint8");
subplot(4, 4, 7);
imshow(myImgGrayMasked);
title('Gray with mask applied');

% Lastly, show the image histogram of the greyscale image after the mask
% has been applied
subplot(4, 4, 8);
imhist(myImgGrayMasked);
title('Image Histogram Grey After Mask');
% NOTE: As we can see from the output, the result is imperfect as the
% output still contains a lot of pixels of the table surface seen in the
% image above the bar of chocolate. We could try to find better thresholds
% but even then the result will be imperfect. Therefore...

% ...quite often in computer vision and image analysis, we need to apply
% multiple algorithms sequentially, rather than having one algorithm that
% does everything we want. Here, it would be helpful to first detect the
% edges of the block of chocolate, to extract just that part as a subimage
% and then to do the above analysis again just on the subimage. We will see
% in coming weeks how we can automate that detection; for now, we simply
% hardcode the top-left and bottom-right coordinates of the block of chocolate.
subImgBlue = myImgBlue(570:1900, 130:3000);
subplot(4, 4, 9);
imshow(subImgBlue);
title('Subimage Chocolate (Blue Channel)');

% Show the image histogram of the blue channel
subplot(4, 4, 10);
imhist(subImgBlue);
title('Histogram Subimage (Blue Channel)');

% Apply the two thresholds again
% Use a comparison and a bitwise AND.
subImgBlueChannelBinaryMask = (subImgBlue >= iLowerThreshold) & ...
                              (subImgBlue <= iUpperThreshold);

% Display the blue channel mask for the subimage
subplot(4, 4, 11);
imshow(subImgBlueChannelBinaryMask);
title('Subimage Blue Channel Mask');

% Apply the binary mask to the blue channel subimage.
subImgBlueMasked = subImgBlue .* cast(subImgBlueChannelBinaryMask, "uint8");
subplot(4, 4, 12);
imshow(subImgBlueMasked);
title('Subimage Blue Ch. with Mask Applied');

% Now let's perform histogram equalisation on the thresholded image
subImgBlueHistEq = histeq(subImgBlue);
subplot(4, 4, 13);
imshow(subImgBlueHistEq);
title('Subimage Blue Ch. Hist. Eq.');

% Display the image histogram after histogram equalisation
subplot(4, 4, 14);
imhist(subImgBlueHistEq);
title('Histogram of Subimage After Hist. Eq.');

% Apply the binary mask to the histogram equalised blue channel subimage
subImgBlueMaskedHistEq = subImgBlueHistEq .* ...
                         cast(subImgBlueChannelBinaryMask, "uint8");
subplot(4, 4, 15);
imshow(uint8(255 * mat2gray(subImgBlueMaskedHistEq))); % Scale pixel values to range [0-255]
title('Subimage After Hist. Eq. and Mask Applied');
% NOTE: As the results show here, it might be better to find new threshold
% values after histogram equalisation. You can explore this on your own.
