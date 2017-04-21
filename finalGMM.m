%%
clear;clc;close all;
GMM = 'GMM.avi';
Obj = VideoWriter(GMM);
writerObj.FrameRate = 30;
open(Obj);

vidIn = 'badformshade.avi';
vidObj = VideoReader(vidIn); 
nFrames = vidObj.NumberOfFrames;
vidHeight = vidObj.Height; 
vidWidth = vidObj.Width;

% Train frames 
foregroundDetector = vision.ForegroundDetector('NumGaussians', 3,...
    'NumTrainingFrames', 130,'LearningRate',.0001);

% Complete background subtraction and Write Video
videoReader = vision.VideoFileReader(vidIn);

% Morphological parameter
seOpen = strel('square', 12);
%  seClose = strel('square', 1);

BBox = zeros(nFrames,4);
for i = 1:nFrames
    frame = read(vidObj,i);
    foreground = step(foregroundDetector, frame);    
%   writeVideo(Obj,double(foreground));    
    IMopen = imopen(foreground,seOpen);
%   IMclose = imclose(foreground,seClose);   

% Blob Bnalysis to Obtain Boundary
    blobAnalysis = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
        'AreaOutputPort', false, 'CentroidOutputPort', false, ...
        'MinimumBlobArea', 4000);
    bbox = step(blobAnalysis, IMopen);
% If there are multiple bounding boxes
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
clc;close all;
% Set Range and Analyze
cutoff= floor(nFrames*(1/3));
tempbox = BBox(cutoff:2*cutoff,3);  
tempboxH = BBox(cutoff:2*cutoff,4);

% Locate Butt
tempmax = find(tempbox == max(tempbox));
endFrameLoc = tempmax+cutoff-1;
vidObj2 = VideoReader(GMM); 

endFrameLoc = endFrameLoc(1,:);
EndFrame = (read(vidObj2,endFrameLoc));
% imshow(uint8(EndFrame));

EndFrameBBox = BBox(endFrameLoc,:);
buttCol = EndFrameBBox(1)+EndFrameBBox(3)-1;
buttRow = find(EndFrame(:,buttCol) > 50,1);

buttCoords = zeros(nFrames,2);

% Create Template Around Butt
buttFrame = rgb2gray(read(vidObj,endFrameLoc));

backTempColStart = round(EndFrameBBox(3)*(1/3)); %first hardcoded the size of the box for our image
backTempColEnd = round(EndFrameBBox(3)*(1/10));
backTempRowStart = round(EndFrameBBox(4)*(3/5));

buttCoords(endFrameLoc,1) = buttCol-backTempColStart;
buttCoords(endFrameLoc,2) = buttRow-backTempRowStart;

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

buttCol = EndFrameBBox(1)+EndFrameBBox(3)-1;
buttRow = find(EndFrame(:,buttCol) > 50,1);

backTemp = buttFrame(buttRow-backTempRowStart:buttRow, ...
           buttCol-backTempColStart:buttCol+backTempColEnd);
       
for m = endFrameLoc+1:2*cutoff
    nextFrame = rgb2gray(read(vidObj,m));
    [dx, dy, matchblock] = templatematching(backTemp,nextFrame,buttCol-backTempColStart,buttRow-backTempRowStart ,1);

    imshow(uint8(matchblock));
    backTemp = matchblock;
    
    buttCol = buttCol+dx;
    buttRow = buttRow+dy;
    
   buttCoords(m,1) = buttCol-backTempColStart;
   buttCoords(m,2) = buttRow-backTempRowStart;
end

%% Pinpointing back as point of interest

buttObj = VideoReader(GMM);
buttPOI = zeros(nFrames,2);

for m = cutoff:2*cutoff
    frame = read(buttObj,m);
    bFrame = frame(buttCoords(m,2):backTempRowStart+buttCoords(m,2),buttCoords(m,1):backTempColStart+buttCoords(m,1)+backTempColEnd);
    imshow(bFrame);
    binaryMatrix = bFrame > 200;
    [sel c] = max(binaryMatrix ~= 0, [], 1);
    binLoc = sel.*c;
    binCol = max(find(binLoc));
    binRow = min(find(binaryMatrix(:,binCol)));
    buttPOI(m,1) = binCol+buttCoords(m,1);
    buttPOI(m,2) = binRow+buttCoords(m,2);
end
%% Testing Curvatures
clc;close all;
GMMObj = VideoReader(GMM); 
fontSize = 20;

% Use Face to Obtain Frame 
testFrame = rgb2gray(read(GMMObj,660));
bwFrame = bwboundaries(testFrame);

BW = testFrame > 128;
[B,L] = bwboundaries(BW,'noholes');
figure;
imshow(label2rgb(L, @jet, [.5 .5 .5]))
hold on;
%initialize size
Bcell = size(B{1},1);
k=1;
for i = 2:size(B,1)
    if size(B{i},1) > Bcell
        k = i;
    end
end
% for k = 1:length(B)
   boundary = B{k};
   plot(boundary(1:floor(length(boundary)),2), ...
   boundary(1:floor(length(boundary)),1), 'r', 'LineWidth', 2)
%end

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


threshold = 1;
cur = curvature > threshold;
Candidates = cur .*curvature;
% figure;plot(Candidates);


indexCurv = find(Candidates);
pointCand = boundary(indexCurv,:);
plot(pointCand(:,2),pointCand(:,1),'x','MarkerSize' ,20);
% 2 = x, 1 = y
% Plot curvature.
figure;plot(curvature, 'b-', 'LineWidth', 1)
grid on;
xlim([1 numberOfPoints]); % Set limits for the x axis.
title('Radius of Curvature', 'FontSize', fontSize);
%% Find standing frame

% Set Range and Analyze
cutoff= floor(nFrames*(1/3));
tempboxH = BBox(cutoff:2*cutoff,4);

% Locate Butt
tempmax = find(tempboxH == max(tempboxH),1);
bendingFrameLoc = tempmax+cutoff-1;
vidObj2 = VideoReader(GMM);

bendingFrameLoc = bendingFrameLoc(1,:);
EndFrame = (read(vidObj2,bendingFrameLoc));

%% HOG 
vidIn = 'badformshade.avi';
vidObj = VideoReader(vidIn); 
hogFrame = rgb2gray(read(vidObj,bendingFrameLoc));
figure;
imshow(hogFrame);
% [featureVector,hogVisualization] = extractHOGFeatures(hogFrame);
% hold on;
% plot(hogVisualization);
% nBlock = length(featureVector)/324;

%Template neck
I = rgb2gray(read(vidObj,650));
hogtemp = I(44-16:44+16,316-16:316+16);
[tempFeature,tempVisualization] = extractHOGFeatures(hogtemp);
% 
% %Template shoulder
% I = rgb2gray(read(vidObj,650));
% hogtemp = I(52:84,304:336);
% figure;
% [tempFeature,tempVisualization] = extractHOGFeatures(hogtemp);

% tempFeature = reshape(tempFeature,36,9);
% imshow(hogtemp);
% hold on;
% plot(tempVisualization);
% ihogCand = 99999999;
% for i = 20000:324:length(featureVector)-324
%     
%     hogCand = (sum(abs(featureVector(i:i+324-1)-tempFeature)));
%     if hogCand<ihogCand
%         ihogCand = hogCand;
%         hogLoc = i;
%     end
% end
ihogCand = 99999;
for i = 1:size(pointCand,1)
    if pointCand(i,1)>16 && pointCand(i,2)>16
    testTemp = hogFrame(pointCand(i,1)-16:pointCand(i,1)+16,...
        pointCand(i,2)-16:pointCand(i,2)+16);
    [testFeature,testVisualization] = extractHOGFeatures(testTemp);
    hogCand = mean(abs(testFeature - tempFeature));
    
    if hogCand < ihogCand
        ihogCand = hogCand;
        hogLoc = i;
    end
    end
end
%pointCand y then x
testTemp = hogFrame(pointCand(hogLoc,1)-16:pointCand(hogLoc,1)+16,...
        pointCand(hogLoc,2)-16:pointCand(hogLoc,2)+16);
    imshow(testTemp);
[testFeature,testVisualization] = extractHOGFeatures(testTemp);hold on
plot(testVisualization);
figure;
imshow(hogtemp);hold on 
plot(tempVisualization);

%% template match around Neck area

neckRow = pointCand(hogLoc,1);
neckCol = pointCand(hogLoc,2);
neckTemp = hogFrame(neckRow-30:neckRow+50,neckCol-30:neckCol+30);

neckCoords = zeros(nFrames,2);
neckCoords(hogLoc,1) = pointCand(hogLoc,1);
neckCoords(hogLoc,2) = pointCand(hogLoc,2);

%matching backwards
for m = bendingFrameLoc-1:-1:cutoff
    prevFrame = rgb2gray(read(vidObj,m));
    [dx, dy, matchblock] = templatematching(neckTemp,prevFrame,neckCol-30,neckRow-30,5);

    neckTemp = matchblock;
    imshow(uint8(matchblock));
    neckRow = neckRow + dy;
    neckCol = neckCol + dx;
    
    neckCoords(m,1) = neckRow - 30;
    neckCoords(m,2) = neckCol - 30;
    
end
%% matching forwards

neckRow = pointCand(hogLoc,1);
neckCol = pointCand(hogLoc,2);
neckTemp = hogFrame(neckRow-30:neckRow+80,neckCol-30:neckCol+20);

for m = bendingFrameLoc+1:2*cutoff
    nextFrame = rgb2gray(read(vidObj,m));
    [dx, dy, matchblock] = templatematching(neckTemp,nextFrame,neckCol-30,neckRow-30,1);

    neckTemp = matchblock;
    imshow(uint8(matchblock));
    
    neckRow = neckRow + dy;
    neckCol = neckCol + dx;
    
    neckCoords(m,1) = neckRow - 30;
    neckCoords(m,2) = neckCol - 30;
    
end
%% Testing to verify hog works
I2 = rgb2gray(read(vidObj,790));
figure;
imshow(I2);
[featureVector2,hogVisualization2] = extractHOGFeatures(I2);
hold on;
plot(hogVisualization2);

tempI2 = I2(100:132,300:332);
figure;
[tempFeature2,tempVisualization2] = extractHOGFeatures(tempI2);
% tempFeature2 = reshape(tempFeature2,36,9);
imshow(tempI2);
hold on;
plot(tempVisualization2);


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