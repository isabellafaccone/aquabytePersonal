numImages = 10;
images1 = cell(1, numImages);
images2 = cell(1, numImages);
left_dir = '/Users/aloksaxena/Documents/aquabyteai/repos/cv_research/alok/notebooks/rectification/data/sf_test_images_large/rectified_left';
right_dir = '/Users/aloksaxena/Documents/aquabyteai/repos/cv_research/alok/notebooks/rectification/data/sf_test_images_large/rectified_right';

left_dir_data = dir(strcat(left_dir, '/*.jpg'));
right_dir_data = dir(strcat(right_dir, '/*.jpg'));
left_files = fullfile({left_dir_data.folder}.', {left_dir_data.name}.');  
right_files = fullfile({right_dir_data.folder}.', {right_dir_data.name}.');

[imagePoints,boardSize,pairsUsed] = detectCheckerboardPoints(left_files,right_files);