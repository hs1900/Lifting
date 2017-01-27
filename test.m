clear;clc;

Obj = VideoWriter('60frames.avi');
writerObj.FrameRate = 60;

vidObj = VideoReader('IMG_0410.mov'); 
nFrames = vidObj.NumberOfFrames;
vidHeight = vidObj.Height; 
vidWidth = vidObj.Width;

% Taking the difference between adjacent frames
% Assuming the foreground is the person of interest
% 
T = 40;
diffT = zeros(vidHeight, vidWidth, 100);


for i = 2 : 100
    frame1 = rgb2gray(read(vidObj, i-1));
    frame2 = rgb2gray(read(vidObj, i));
    diff = abs(double(frame1) - double(frame2));
    diffT(:,:,i-1) = (diff>T);
%   writeVideo(Obj,double((diffT(:,:,i-1))));
end

%downsampling
downDiffT = diffT(1:2:end,1:2:end,:);

open(Obj);
for i = 2: 100
    writeVideo(Obj,double((diffT(:,:,i-1))));
end    


close(Obj);
