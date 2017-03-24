%% GMM
clear;clc;close all;

Obj = VideoWriter('GMMbox.avi');
writerObj.FrameRate = 30;
open(Obj);

vidObj = VideoReader('badformshade.m4v'); 
nFrames = vidObj.NumberOfFrames;
vidHeight = vidObj.Height; 
vidWidth = vidObj.Width;

% Train frames with no foreground
foregroundDetector = vision.ForegroundDetector('NumGaussians', 3,'NumTrainingFrames', 130,'LearningRate',.0001);
% Complete background subtraction
videoReader = vision.VideoFileReader('badformshade.m4v');
seOpen = strel('square', 12);
%  seClose = strel('square', 1);
BBox = zeros(nFrames,4);
for i = 1:nFrames
    frame = step(videoReader); % read the next video frame
    foreground = step(foregroundDetector, frame);    
%   writeVideo(Obj,double(foreground));    
    IMopen = imopen(foreground,seOpen);
%   IMclose = imclose(foreground,seClose);   

% blob analysis
blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
    'AreaOutputPort', false, 'CentroidOutputPort', false, ...
    'MinimumBlobArea', 4000);
bbox = step(blobAnalysis, IMopen);

index = find(max(bbox(:,3) .*bbox(:,4)));
if size(bbox) > [0 3]
BBox(i,:) = bbox(index,:);
end

result = insertShape( 255*uint8(IMopen), 'Rectangle', BBox(i,:), 'Color', 'green');

IMopen = result;

    writeVideo(Obj,(IMopen));
end

close(Obj);

%% Matching butt
cutoff= floor(nFrames*(1/3));
tempbox = BBox(cutoff:2*cutoff,3);
tempmax = find(tempbox == max(tempbox));
endFrameLoc = tempmax+cutoff;
vidObj2 = VideoReader('GMMbox.avi'); 

EndFrame = read(Obj2,endFrameLoc);
imshow(uint8(EndFrame));
%% Edge
edgeObj = VideoWriter('edgeGMM.avi');
GMMObj = VideoReader('GMM.avi'); 
open(edgeObj);
for i = 1:nFrames
    frame = rgb2gray(read(GMMObj,i)); % read the next video frame
    edgeFrame = edge(frame, 'canny',.5,2);
    writeVideo(edgeObj,double(edgeFrame));
end
close(edgeObj);
edgeObj = VideoReader('edgeGMM.avi');
testFrame = rgb2gray(read(edgeObj,780));
imshow(testFrame);
%% testing
clc;
GMMObj = VideoReader('GMM.avi'); 

testFrame = rgb2gray(read(GMMObj,780));
bwFrame = bwboundaries(testFrame);

BW = testFrame > 128;
[B,L] = bwboundaries(BW,'noholes');
figure;
imshow(label2rgb(L, @jet, [.5 .5 .5]))
hold on;
for k = 1:length(B)
   boundary = B{k};
   plot(boundary(:,2), boundary(:,1), 'w', 'LineWidth', 2)
end
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
figure; imshow(result); title('Detected');
%% Process
videoPlayer = vision.VideoPlayer('Name', 'Detected');
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
