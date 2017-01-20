vidObj = VideoWriter('IMG_0410.mov'); 
nFrames = vidObj.NumberOfFrames;
vidHeight = vidObj.Height; 
vidWidth = vidObj.Width;

mov(1:nFrames)= struct('cdata', zeros(vidHeight, vidWidth, 3, 'uint8'))
for i = 1 : nFrames
  mov(i).cdata = read(vidObj, k); 
end
