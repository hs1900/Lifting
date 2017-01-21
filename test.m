vidObj = VideoWriter('IMG_0410.mov'); 
nFrames = vidObj.NumberOfFrames;
vidHeight = vidObj.Height; 
vidWidth = vidObj.Width;

mov(1:nFrames)= struct('cdata', zeros(vidHeight, vidWidth, 3, 'uint8'))
for i = 1 : nFrames
  mov(i).cdata = read(vidObj, k); 
end

% Not sure how I would detect the corners of the block: x0, y0
% R = search range, template = the part to track
function [dx, dy, matchblock] = EMBA(template, img, x0, y0, R)

[H,W] = size(img);
[tempH, tempW] = size(template);
maxerror = tempH * tempW * 255;
dx = 0; dy = 0;

for(k = max(1, x0-R) : min(W-tempW, x0+R)
  for(l = max(1, y0-R) : min(H=tempH, y0+R)
  
    block = img(l:l+tempH-1, k:k+tempW-1);
    error = sum(sum(abs(template-block)));
    if (error<maxerror)
      maxerror = error;
      x0 = k; y0 = l; matchblock = block;
    end
  end
end
    
