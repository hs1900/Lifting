%% GMM
clear;clc;close all;
GMM = 'GMMv2.avi';
Obj = VideoWriter(GMM);
writerObj.FrameRate = 30;
open(Obj);

vidIn = 'badformshade.avi';
vidObj = VideoReader(vidIn); 
nFrames = vidObj.NumberOfFrames;
vidHeight = vidObj.Height; 
vidWidth = vidObj.Width;

% Train frames 
foregroundDetector = vision.ForegroundDetector('NumGaussians', 3,'NumTrainingFrames', 130,'LearningRate',.0001);

% Complete background subtraction and Write Video
%videoReader = vision.VideoFileReader(vidIn);

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
cutoff= floor(nFrames*(1/3)); %processing only the first-second third of the video, this number can be changed

bendingBBox = BBox(cutoff:2*cutoff,3); %locating the bbox with largest width
standingBBox = BBox(cutoff:2*cutoff,4); % locating the bbox with largest height

tempmaxH = find(standingBBox == max(standingBBox),1);
standingFrameLoc = tempmaxH + cutoff -1; %frame location where person is standing
%%
 % Locate Butt
tempmaxW = find(bendingBBox == max(bendingBBox),1);
bendingFrameLoc = tempmaxW+cutoff-1; %frame location where person is bending
vidObj2 = VideoReader(GMM); 

bendingFrameLoc = bendingFrameLoc(1,:);
BendingFrame = (read(vidObj2,bendingFrameLoc));
% imshow(uint8(EndFrame));

BendingFrameBBox = BBox(bendingFrameLoc,:);

%location of right most point of bounding box with person bending down
buttCol = BendingFrameBBox(1)+BendingFrameBBox(3)-1; 
buttRow = find(BendingFrame(:,buttCol) > 50,1);


buttCoordsTemp = zeros(nFrames,2);

% Create Template Around Butt
buttFrame = rgb2gray(read(vidObj,bendingFrameLoc));

%dimensions for the template - first hardcoded the size of the box for our image
buttTempColStart = round(BendingFrameBBox(3)*(1/3)); 
buttTempColEnd = round(BendingFrameBBox(3)*(1/10));
buttTempRowStart = round(BendingFrameBBox(4)*(3/5));

buttCoordsTemp(bendingFrameLoc,1) = buttCol-buttTempColStart;
buttCoordsTemp(bendingFrameLoc,2) = buttRow-buttTempRowStart;

% Create New Template for Back
backTemp = buttFrame(buttRow-buttTempRowStart:buttRow, ...
           buttCol-buttTempColStart:buttCol+buttTempColEnd);

 %Template Match Lower Back
for m = bendingFrameLoc-1:-1:cutoff
    prevFrame = rgb2gray(read(vidObj,m));
    [dx, dy, matchblock] = templatematching(backTemp,prevFrame,buttCol-buttTempColStart,buttRow-buttTempRowStart ,3);

    imshow(uint8(matchblock));
    backTemp = matchblock;
    
    buttCol = buttCol+dx;
    buttRow = buttRow+dy;
    
    buttCoordsTemp(m,1) = buttCol-buttTempColStart;
    buttCoordsTemp(m,2) = buttRow-buttTempRowStart;
end

%% Matching Lower Back Foward

buttCol = BendingFrameBBox(1)+BendingFrameBBox(3)-1;
buttRow = find(BendingFrame(:,buttCol) > 50,1);

backTemp = buttFrame(buttRow-buttTempRowStart:buttRow, ...
           buttCol-buttTempColStart:buttCol+buttTempColEnd);
       
for m = bendingFrameLoc+1:2*cutoff
    nextFrame = rgb2gray(read(vidObj,m));
    [dx, dy, matchblock] = templatematching(backTemp,nextFrame,buttCol-buttTempColStart,buttRow-buttTempRowStart ,1);

    imshow(uint8(matchblock));
    backTemp = matchblock;
    
    buttCol = buttCol+dx;
    buttRow = buttRow+dy;
    
   buttCoordsTemp(m,1) = buttCol-buttTempColStart;
   buttCoordsTemp(m,2) = buttRow-buttTempRowStart;
end

%% Pinpointing back as point of interest

%after obtaining the locations for the back template, we find the rightmost point in each frame
buttObj = VideoReader(GMM);
buttPOI = zeros(nFrames,2);

for m = cutoff:2*cutoff
    frame = read(buttObj,m);
    bFrame = frame(buttCoordsTemp(m,2):buttTempRowStart+buttCoordsTemp(m,2),buttCoordsTemp(m,1):buttTempColStart+buttCoordsTemp(m,1)+buttTempColEnd);
    imshow(bFrame);
    binaryMatrix = bFrame > 200;
    
    [sel c] = max(binaryMatrix ~= 0, [], 1);
    binLoc = sel.*c;
    binCol = max(find(binLoc));
    binRow = min(find(binaryMatrix(:,binCol)));
    
    buttPOI(m,1) = binCol+buttCoordsTemp(m,1);
    buttPOI(m,2) = binRow+buttCoordsTemp(m,2);
end