function [h, s, v] = colourAnalysis(image)
    % Convert the input image to HSV color space
    hsvImage = rgb2hsv(image);

    % Extract the hue, saturation, and value channels
    h = hsvImage(:,:,1);
    s = hsvImage(:,:,2);
    v = hsvImage(:,:,3);
end