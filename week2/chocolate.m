clear variables;
close all;
clc;

myImg = imread("chocolate_original.jpg");
myImg = imresize(myImg,[480 640]);
[r,g,b] = imsplit(myImg);

figure('Name','Splitted Image','NumberTitle','off');
subplot(3,4,1);
imshow(r);
title("red");

figure(1);
subplot(3,4,2);
imshow(g);
title("green");

figure(1);
subplot(3,4,3);
imshow(b);
title("blue");

choc_gray=rgb2gray(myImg);
figure(2);
imshow(choc_gray);

level=graythresh(choc_gray);
BW = imbinarize(choc_gray,level);
figure(3);
imshowpair(choc_gray,BW,'montage');

% Threshold image
for iI = 0.0:0.01:1.0
    thresholdedImg = imbinarize(choc_gray, iI);
    figure(4);
    imshow(thresholdedImg);
    title(string(iI));
    pause(0.05);
end
