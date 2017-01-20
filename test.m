vidObj = VideoWriter('IMG_0410.mov'); 
nFrames = vidObj.NumberOfFrames;
vidHeight = vidObj.Height; 
vidWidth = vidObj.Width;

mov(1:nFrames)= struct('cdata', zeros(vidHeight, vidWidth, 3, 'uint8'))
for i = 1 : nFrames
  mov(i).cdata = read(vidObj, k); 
end

% Taking the difference between adjacent frames
% Assuming the foreground is the person of interest
frame1(1) = mov(1);
for i = 2 : nFrame
 diff = abs(frame1(i-1) – mov(i));
 if diff >=64
  th = 1;
  else th = 0;
 frame1(i) = mov(i);
end
