%% Background Sutraction
clear;clc;

Obj = VideoWriter('bS.avi');
writerObj.FrameRate = 30;
open(Obj);
% vision
vidObj = VideoReader('Test2.m4v'); 
nFrames = vidObj.NumberOfFrames;
vidHeight = vidObj.Height; 
vidWidth = vidObj.Width;

% Using the beginning frame as the background 
% Assuming the foreground is the person of interest

T = 10;
frameBG = 15;
frameBeg = 1;
frameEnd = 600;

% diffT = zeros(vidHeight, vidWidth, frameEnd-frameBeg);
BG = double(rgb2gray(read(vidObj,frameBG)));
% for averaging initial frames
% for i = 2 : frameBG
%     frame = double(rgb2gray(read(vidObj,i)));
%     BG = frame+BG;
% end
% BG = BG/frameBG;

% Morphological
se = strel('square', 3);


for i = frameBeg : frameEnd
  diff = abs(double(rgb2gray(read(vidObj, i))) - BG);
%   diffT(:,:,i-frameBeg+1) = (diff>T);
  diffT = (diff>T);
  
  writeVideo(Obj,double((diffT)));
  
%   filteredForeground = imopen(diffT, se);
%   writeVideo(Obj,double((filteredForeground)));

end


close(Obj);