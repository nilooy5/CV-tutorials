rgbImage = imread('images/robocup_image1.jpg');
[hue1, saturation1, value1] = colourAnalysis(rgbImage);
figure(1);
subplot(2,3,2);
imshow(rgbImage);
title("original image");


figure(1);
subplot(2, 3, 4);
imshow(hue1);
title('hue');

subplot(2, 3, 5);
imshow(saturation1);
title('saturation');

subplot(2, 3, 6);
imshow(value1);
title('value');