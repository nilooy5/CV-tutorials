% Background subtraction example on thermal images
% Author: Roland Goecke, (c) 2018

clear variables;    % This is similar to 'clear all' but more efficient.
close all;
clc;

% Load images *** Change file path to your current settings ***
backgroundImg = imread('Images\DINGO3_Background.jpeg');
dingoImg = imread('Images\DINGO3_Frame0.jpeg');

% Display images
figure(1);
imshow(backgroundImg);

figure(2);
imshow(dingoImg);

separateBackground(dingoImg, backgroundImg);
demo();

function separateBackground(im1,im2)
    % Compute difference image
    % (Literally, subtract the background image from the current image.)
    diffImg = im1 - im2;

    % Find the minimum pixel value. Divide by 255 as we will need this value
    % to be in the range of 0 to 1.
    minPixelValue = double(min(min(diffImg)))/255.0;
    disp('Min Pixel Value for Red, Green, Blue channels:');
    disp(minPixelValue);

    % Find the maximum pixel value. Divide by 255 as we will need this value
    % to be in the range of 0 to 1.
    maxPixelValue = double(max(max(diffImg)))/255.0;
    disp('Max Pixel Value for Red, Green, Blue channels:');
    disp(maxPixelValue);

    % Display difference image without rescaling
    figure(3);
    imshow(diffImg);

    % Display difference image with rescaling
    rescaledDiffImg = imadjust(diffImg, ...
                        [minPixelValue(1) minPixelValue(2) minPixelValue(3); ...
                        maxPixelValue(1) maxPixelValue(2) maxPixelValue(3)], ...
                    []);
    figure(4);
    imshow(rescaledDiffImg);

    % Convert into greyscale image
    greyDiffImg = rgb2gray(rescaledDiffImg);
    figure(5);
    imshow(greyDiffImg);

    % Threshold image - Experiment with different threshold values!
    thresholdedImg = imbinarize(greyDiffImg, 0.1);
    figure(6);
    imshow(thresholdedImg);

    greyDiffImg = rgb2gray(rescaledDiffImg);
    BW_dingo = imbinarize(greyDiffImg, 0.1);

    se = strel('sphere', 5);
    dilated_dingo = imdilate(BW_dingo,se);


    se = strel('sphere', 4);
    eroded_dingo = imerode(dilated_dingo,se);

    figure(6);
    imshow(dilated_dingo);

    figure(7);
    imshow(eroded_dingo);

    cut_dingo = cast(eroded_dingo, 'uint8') .* im1;

    figure(8);
    imshow(cut_dingo);
end



function demo()
    disp('Hello world')
end

% q: what does the ... in matlab mean?

% a: The ... is a placeholder for the rest of the arguments. It is used
% when you don't know how many arguments you will need. For example, if
% you want to pass in a variable number of arguments, you can use the ...
% to indicate that you don't know how many arguments you will need.



% q: write a demo function in matlab

% a: function demo()
%     disp('Hello World');
% end
