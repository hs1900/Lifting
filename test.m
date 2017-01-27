clear;clc;

Obj = VideoWriter('60frames.avi');
writerObj.FrameRate = 30;

vidObj = VideoReader('IMG_0410.mov'); 
nFrames = vidObj.NumberOfFrames;
vidHeight = vidObj.Height; 
vidWidth = vidObj.Width;

% Taking the difference between adjacent frames
% Assuming the foreground is the person of interest
% 
T = 20;
frameBeg = 530;
selFrame = 100;
diffT = zeros(vidHeight, vidWidth, selFrame);


for i = frameBeg : selFrame+frameBeg-1
    frame1 = rgb2gray(read(vidObj, i));
    frame2 = rgb2gray(read(vidObj, i+3));
    diff = abs(double(frame1) - double(frame2));
    diffT(:,:,i+1-frameBeg) = (diff>T);
%   writeVideo(Obj,double((diffT(:,:,i-1))));
end

%downsampling
downDiffT = diffT(1:2:end,1:2:end,:);

open(Obj);
for i = 1: selFrame
    writeVideo(Obj,double((downDiffT(:,:,i))));
end    

close(Obj);

% Need a low T for accurate representation
% Filtering noise created from slight changes
