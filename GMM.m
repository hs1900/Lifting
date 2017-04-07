%% GMM
clear;clc;close all;
GMM = 'GMM.avi';
Obj = VideoWriter(GMM);
writerObj.FrameRate = 30;
open(Obj);

vidIn = 'badformshade.m4v';
vidObj = VideoReader(vidIn); 
nFrames = vidObj.NumberOfFrames;
vidHeight = vidObj.Height; 
vidWidth = vidObj.Width;

% Train frames 
foregroundDetector = vision.ForegroundDetector('NumGaussians', 3,'NumTrainingFrames', 130,'LearningRate',.0001);

% Complete background subtraction and Write Video
videoReader = vision.VideoFileReader(vidIn);

% Morphological parameter
seOpen = strel('square', 12);
%  seClose = strel('square', 1);

BBox = zeros(nFrames,4);
for i = 1:nFrames
    frame = step(videoReader); % read the next video frame
    foreground = step(foregroundDetector, frame);    
%   writeVideo(Obj,double(foreground));    
    IMopen = imopen(foreground,seOpen);
%   IMclose = imclose(foreground,seClose);   

% Blob Bnalysis to Obtain Boundary
    blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
        'AreaOutputPort', false, 'CentroidOutputPort', false, ...
        'MinimumBlobArea', 4000);
    bbox = step(blobAnalysis, IMopen);

    index = find(max(bbox(:,3) .*bbox(:,4)));
    if size(bbox) > [0 3]
        BBox(i,:) = bbox(index,:);
    end
    
% Can Insert Box for Visual Confirmation
%     result = insertShape( 255*uint8(IMopen), 'Rectangle', BBox(i,:), 'Color', 'green');
%     IMopen = result;

    writeVideo(Obj,255*uint8(IMopen));
end

close(Obj);

%% Matching Lower Back

% Set Range and Analyze
cutoff= floor(nFrames*(1/3));
tempbox = BBox(cutoff:2*cutoff,3);

% Locate Butt
tempmax = find(tempbox == max(tempbox));
endFrameLoc = tempmax+cutoff-1;
vidObj2 = VideoReader(GMM); 

EndFrame = (read(vidObj2,endFrameLoc));
% imshow(uint8(EndFrame));

EndFrameBBox = BBox(endFrameLoc,:);
buttCol = EndFrameBBox(1)+EndFrameBBox(3)-1;
buttRow = find(EndFrame(:,buttCol) > 50,1);

% Create Template Around Butt
buttFrame = rgb2gray(read(vidObj,endFrameLoc));

backTempColStart = round(EndFrameBBox(3)*(1/3)); %first hardcoded the size of the box for our image
backTempColEnd = round(EndFrameBBox(3)*(1/15));
backTempRowStart = round(EndFrameBBox(4)*(4/5));

% buttRow-80, buttRow                         buttCol-50:buttCol+10
% Create New Template for Back
backTemp = buttFrame(buttRow-backTempRowStart:buttRow, ...
           buttCol-backTempColStart:buttCol+backTempColEnd);

 %Template Match Lower Back
for m = endFrameLoc-1:-1:cutoff
    prevFrame = rgb2gray(read(vidObj,m));
    [dx, dy, matchblock] = templatematching(backTemp,prevFrame,buttCol-backTempColStart,buttRow-backTempRowStart ,3);

    imshow(uint8(matchblock));
    backTemp = matchblock;
    
    buttCol = buttCol+dx;
    buttRow = buttRow+dy;
    
    buttCoords(m,1) = buttCol-backTempColStart;
    buttCoords(m,2) = buttRow-backTempRowStart;
end

%% Matching Lower Back Foward


%% Testing Curvatures
clc;close all;
GMMObj = VideoReader(GMM); 
fontSize = 20;

% Use Face to Obtain Frame 
testFrame = rgb2gray(read(GMMObj,780));
bwFrame = bwboundaries(testFrame);

BW = testFrame > 128;
[B,L] = bwboundaries(BW,'noholes');
figure;
imshow(label2rgb(L, @jet, [.5 .5 .5]))
hold on;
for k = 1:length(B)
   boundary = B{k};
   plot(boundary(1:floor(length(boundary)/30),2), ...
   boundary(1:floor(length(boundary)/30),1), 'r', 'LineWidth', 2)
end

%find radius and curvature
numberOfPoints = length(boundary);
curvature = zeros(1, numberOfPoints);
for t = 1 : numberOfPoints
	if t == 1
		index1 = numberOfPoints;
		index2 = t;
		index3 = t + 1;
	elseif t >= numberOfPoints
		index1 = t-1;
		index2 = t;
		index3 = 1;
	else
		index1 = t-1;
		index2 = t;
		index3 = t + 1;
	end
	% Get the 3 points.
	x1 = boundary(index1,2);
	y1 = boundary(index1,1);
	x2 = boundary(index2,2);
	y2 = boundary(index2,1);
	x3 = boundary(index3,2);
	y3 = boundary(index3,1);

	curvature(t) = 2*abs((x2-x1).*(y3-y1)-(x3-x1).*(y2-y1)) ./ ...
	sqrt(((x2-x1).^2+(y2-y1).^2)*((x3-x1).^2+(y3-y1).^2)*((x3-x2).^2+(y3-y2).^2));
end

% Plot curvature.
figure;plot(curvature, 'b-', 'LineWidth', 1)
grid on;
xlim([1 numberOfPoints]); % Set limits for the x axis.
title('Radius of Curvature', 'FontSize', fontSize);

threshold = 1;
cur = curvature > threshold;
Candidates = cur .*curvature;
figure;plot(Candidates);
%% HOG 
peopleDetector = vision.PeopleDetector;
y = step(peopleDetector,endFrameLoc) ;
figure;imshow(y);
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
%% References Below
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
