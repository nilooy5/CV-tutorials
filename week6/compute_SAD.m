% Function to compute the SAD similarity measure
function [SAD_values, tmpHeightHalf, tmpWidthHalf, imgSize] = compute_SAD(inputImg,tmpImg)
    imgSize = size(inputImg);
    tmpSize = size(tmpImg);
    tmpHeightHalf = tmpSize(1)/2;
%     tmpHeightHalf = cast(tmpHeightHalf,"uint8");
    tmpWidthHalf = tmpSize(2)/2;
%     tmpWidthHalf = cast(tmpWidthHalf, "uint8");
    SAD_values = zeros(imgSize); % this will be data type double
    % Loop over possible template positions (nothing extending over
    % edges!) and calculate the SAD value at the centre pixel
    for iRow = tmpHeightHalf : imgSize(1)-tmpHeightHalf
        for iCol = tmpWidthHalf : imgSize(2)-tmpWidthHalf
            SAD_values(iRow, iCol) = sum(abs(inputImg(1+iRow-tmpHeightHalf:iRow+tmpHeightHalf, 1+iCol-tmpWidthHalf:iCol+tmpWidthHalf) - tmpImg), 'all');
        end
    end