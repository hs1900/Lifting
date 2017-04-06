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
endFrameLoc = tempmax+cutoff-1;
vidObj2 = VideoReader('GMM.avi'); 

EndFrame = (read(vidObj2,endFrameLoc));
imshow(uint8(EndFrame));

EndFrameBBox = BBox(endFrameLoc,:);
buttCol = EndFrameBBox(1)+EndFrameBBox(3)-1;
buttRow = find(EndFrame(:,buttCol) > 50,1);

% Create template around butt
buttFrame = rgb2gray(read(vidObj,endFrameLoc));
% buttTemp = buttFrame(buttRow-30:buttRow+30,buttCol-60:buttCol);

bibuttTemp = EndFrame(buttRow-25:buttRow+20,buttCol-40:buttCol,1);

% Create new template closer to the lower back
buttCol=buttCol-40;
buttRow = buttRow -25 + find(bibuttTemp(:,1) > 50,1);
backTemp = buttFrame(buttRow-40:buttRow,buttCol-10:buttCol+20);
% tempxy = zeros(nFrames,2); tempxy(endFrameLoc,:) = [buttRow-8,buttCol-8] ;
% offset tempxy by 8
 
for m = endFrameLoc-1:-1:100
    prevFrame = rgb2gray(read(vidObj,m));
%     prevBox = prevFrame(BBox(m,2):BBox(m,2)+BBox(m,4),BBox(m,1):BBox(m,1)+BBox(m,3));
%     [tempRow,tempCol] = size(buttTemp);
    [dx, dy, matchblock] = templatematching(backTemp,prevFrame,buttCol-10,buttRow-40 ,1);
%     check to see if matchblock is correct
     if mod(m,2)==0     
         figure;imshow(uint8(matchblock));
     end
    backTemp = matchblock;
%     tempxy( -m+endFrameLoc,:) = [dx +tempxy(-m+endFrameLoc-1,1), dy+tempxy(-m+endFrameLoc-1,2)];
    buttCol = buttCol+dx;
    buttRow = buttRow+dy;
end

%% Butt Match Foward
cutoff= floor(nFrames*(1/3));
tempbox = BBox(cutoff:2*cutoff,3);
tempmax = find(tempbox == max(tempbox));
endFrameLoc = tempmax+cutoff-1;
vidObj2 = VideoReader('GMM.avi'); 

EndFrame = (read(vidObj2,endFrameLoc));
imshow(uint8(EndFrame));

EndFrameBBox = BBox(endFrameLoc,:);
buttCol = EndFrameBBox(1)+EndFrameBBox(3)-1;
buttRow = find(EndFrame(:,buttCol) > 50,1);

% Create template around butt
buttFrame = rgb2gray(read(vidObj2,endFrameLoc));
buttTemp = buttFrame(buttRow-25:buttRow+20,buttCol-40:buttCol);

% Create new template closer to the lower back
% buttCol=buttCol-40;
% buttRow = buttRow -25 + find(bibuttTemp(:,1) > 50,1);
% backTemp = buttFrame(buttRow-40:buttRow,buttCol-10:buttCol+20);
 
for m = endFrameLoc+1:1:850
    nextFrame = rgb2gray(read(vidObj,m));
    [dx, dy, matchblock] = templatematching(buttTemp,nextFrame,buttCol-40,buttRow-25 ,1);
%     check to see if matchblock is correct 
  if mod(m,2)==0     
         figure;imshow(uint8(matchblock));
  end      
    buttTemp = matchblock;
%     tempxy( -m+endFrameLoc,:) = [dx +tempxy(-m+endFrameLoc-1,1), dy+tempxy(-m+endFrameLoc-1,2)];
    buttCol = buttCol+dx;
    buttRow = buttRow+dy;
end
%% Testing EBMA
m = 25;
prevFrame = rgb2gray(read(vidObj,m));
template = buttTemp; img = prevframe; x0 = buttCol-20; y0 = buttRow - 50; R = 2;

[H,W]=size(img);
[BH,BW]=size(template);
maxerror=BH*BW*255;
for (k=max(1,x0-R):min(W-BW,x0+R))
    for (l=max(1,y0-R):min(H-BH,y0+R))
        block=img(l:l+BH-1,k:k+BW-1);
        error=sum(sum(abs(block-template)));
       if (error<maxerror)
             dx=k-x0;dy=l-y0;matchblock=block;
             maxerror=error;
       end
    end
end
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
%% Testing Curvatures
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
   plot(boundary(1:floor(length(boundary)/30),2), ...
   boundary(1:floor(length(boundary)/30),1), 'b', 'LineWidth', 2)
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
	% Now call Roger's formula:
	% http://www.mathworks.com/matlabcentral/answers/57194#answer_69185
	curvature(t) = 2*abs((x2-x1).*(y3-y1)-(x3-x1).*(y2-y1)) ./ ...
	sqrt(((x2-x1).^2+(y2-y1).^2)*((x3-x1).^2+(y3-y1).^2)*((x3-x2).^2+(y3-y2).^2));
end

% Plot curvature.
figure;
plot(curvature, 'b-', 'LineWidth', 1)
grid on;
xlim([1 numberOfPoints]); % Set limits for the x axis.
title('Radius of Curvature', 'FontSize', fontSize);

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
