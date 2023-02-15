clear variables;
close all;
clc;

myImg = imread("chocolate_original.jpg");
myImg = imresize(myImg,[480 640]);
[r,g,b] = imsplit(myImg);

choc_gray=rgb2gray(myImg);
figure(1);
imshow(b);

% level=graythresh(choc_gray);
% BW = imbinarize(choc_gray,level);
% figure(3);
% imshowpair(choc_gray,BW,'montage');

% figure(2);
% imhist(choc_gray);
figure(2);
imhist(b);

t1 = imbinarize(b,34/255);
figure(3);
imshow(t1)

t2 = imbinarize(b,80/255);
figure(4);
imshow(t2)

best_thresh = 80/255-34/255;
fprintf("%f",best_thresh);
t3 = imbinarize(choc_gray,best_thresh);
figure(5);
imshow(t3)