myImg=imread("peppers.png")
myImgSize=size(myImg)
numrows=myImgSize(1)/4
numcols=myImgSize(2)/4
quarterSizedImg=imresize(myImg,0.25)
quarterSized2=imresize(myImg,[numrows numcols])
figure, imshow(quarterSizedImg)

figure, imshow(quarterSized2)
[r,g,b] = imsplit(quarterSized2)
figure, imshow(r)
figure, imshow(g)
figure, imshow(b)