%% GMM
clear;clc;close all;

Obj = VideoWriter('GMM.avi');
writerObj.FrameRate = 30;
open(Obj);

vidObj = VideoReader('badformshade.m4v'); 
nFrames = vidObj.NumberOfFrames;
vidHeight = vidObj.Height; 
vidWidth = vidObj.Width;
%FGvid = zeros(vidHeight,vidWidth,100);
% Start with first 10 frames
foregroundDetector = vision.ForegroundDetector('NumGaussians', 3,'NumTrainingFrames', 20,'LearningRate',.0001);
% Finish with the first 600 frames
videoReader = vision.VideoFileReader('badformshade.m4v');
for i = 1:700
    frame = step(videoReader); % read the next video frame
    foreground = step(foregroundDetector, frame);
    
    writeVideo(Obj,double(foreground));
    
%     se = strel('square', 3);
%     filteredForeground = imopen(foreground, se);
%     writeVideo(Obj,double(filteredForeground));
end
close(Obj);
figure; imshow(frame); title('Video Frame');
figure; imshow(foreground); title('Foreground');
%% Morphological
se = strel('square', 3);
filteredForeground = imopen(foreground, se);
figure; imshow(filteredForeground); title('Clean Foreground');
close(Obj);
%% Box
% Reject blobs less than 150 pixels
blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', false, 'CentroidOutputPort', false, ...
    'MinimumBlobArea', 5000);
bbox = step(blobAnalysis, filteredForeground);
% Create box 
result = insertShape(frame, 'Rectangle', bbox, 'Color', 'green');

numCars = size(bbox, 1);
result = insertText(result, [10 10], numCars, 'BoxOpacity', 1, ...
    'FontSize', 14);
figure; imshow(result); title('Detected Cars');
%% Process
videoPlayer = vision.VideoPlayer('Name', 'Detected Cars');
videoPlayer.Position(3:4) = [650,400];  % window size: [width, height]
se = strel('square', 3); % morphological filter for noise removal

while ~isDone(videoReader)
    
    frame = step(videoReader); % read the next video frame
    
    % Detect the foreground in the current video frame
    foreground = step(foregroundDetector, frame);
    
    % Use morphological opening to remove noise in the foreground
    filteredForeground = imopen(foreground, se);
    
    % Detect the connected components with the specified minimum area, and
    % compute their bounding boxes
    bbox = step(blobAnalysis, filteredForeground);

    % Draw bounding boxes around the detected cars
    result = insertShape(frame, 'Rectangle', bbox, 'Color', 'green');

    % Display the number of cars found in the video frame
    numCars = size(bbox, 1);
    result = insertText(result, [10 10], numCars, 'BoxOpacity', 1, ...
        'FontSize', 14);

    step(videoPlayer, result);  % display the results
end

release(videoReader); % close the video file
