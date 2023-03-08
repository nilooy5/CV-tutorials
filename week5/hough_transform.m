clear variables;    % This is similar to 'clear all' but more efficient.
close all;
clc;

% Load image
myImgBarcode = imread('images\barcode.jpg');
figure(1);
subplot(2, 4, 1);
imshow(myImgBarcode);
title('Original Image');

barcodeGray = rgb2gray(myImgBarcode);
figure(1);
subplot(2, 4, 2);
imshow(barcodeGray);
title('greyscale image');

barcodeEdgeCanny = edge(barcodeGray,"Canny");
subplot(2, 4, 3);
imshow(barcodeEdgeCanny);
title('Canny');

[H, T, R] = hough(barcodeEdgeCanny,'RhoResolution', 0.5, 'Theta', -90:0.5:89);

% Display the Hough matrix (rho and theta value pairs)
subplot(2,3,4);
imshow(imadjust(mat2gray(H)),...
    'XData',T,...
    'YData',R,...
    'InitialMagnification','fit');
title('Hough Transform');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
colormap(hot);

P = houghpeaks(H, 10);
plot(T(P(:,2)), R(P(:,1)), 's', 'color', 'green');

lines = houghlines(barcodeEdgeCanny, T, R, P, 'FillGap', 5, 'MinLength',7);
subplot(2, 3, 5);
imshow(myImgBarcode), hold on;
max_len = 0;
for k = 1:length(lines)
    xy = [lines(k).point1; lines(k).point2];
    plot(xy(:,1),xy(:,2), 'LineWidth', 2, 'Color', 'green');
    % Plot beginnings and ends of lines
    plot(xy(1,1), xy(1,2), 'x', 'LineWidth', 2, 'Color', 'yellow');
    plot(xy(2,1), xy(2,2), 'x', 'LineWidth', 2, 'Color', 'red');
    % Determine the endpoints of the longest line segment
    len = norm(lines(k).point1 - lines(k).point2);
    if ( len > max_len)
        max_len = len;
        xy_long = xy;
    end
end
hold off;
title('Visualisation of Line Segments');
