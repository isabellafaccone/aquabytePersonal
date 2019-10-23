I1 = imread('/Users/aloksaxena/Documents/aquabyteai/repos/cv_research/alok/notebooks/rectification/data/images/left_images/Image__2019-01-30__04-47-32_fish1_leftcamera_lefside.jpg');
I2 = imread('/Users/aloksaxena/Documents/aquabyteai/repos/cv_research/alok/notebooks/rectification/data/images/right_images/Image__2019-01-30__04-51-28_fish1_rightcamera_lefside.jpg');

[J1_valid,J2_valid] = rectifyStereoImages(I1,I2,stereoParams, ...
  'OutputView','full');

imwrite(J1_valid, '~/Desktop/left.jpg')
imwrite(J2_valid, '~/Desktop/right.jpg')

figure;

% before rectification
subplot(2,2,1);
imshow(I1);
hold on;
for i = 1:100:3000
    plot([0, 4095], [i, i]);
    hold on;
end


subplot(2,2,2);
imshow(I2);
hold on;
for i = 1:100:3000
    plot([0, 4095], [i, i]);
    hold on;
end


% after rectification
subplot(2,2,3);
imshow(J1_valid);
hold on;
for i = 1:100:3000
    plot([0, 4095], [i, i]);
    hold on;
end


subplot(2,2,4);
imshow(J2_valid);
hold on;
for i = 1:100:3000
    plot([0, 4095], [i, i]);
    hold on;
end
