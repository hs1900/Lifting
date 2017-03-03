%  Read an image stored in row format, row by row, each pixel represented
%  by an unsigned character
% 
%  Usage : img=bmpread('filename', width, height);
%  Yao Wang, 2014/11/4
% 

function [xm,ym]=EBMADP(template,img,x0,y0,Rx,Ry)

[H,W]=size(img);
[BH,BW]=size(template);
maxerror=BH*BW*255;
dx=0;dy=0;
xm = 0;
ym = 0;
for (k=max(1,x0-Rx):min(W-BW,x0+Rx))
    for (l=max(1,y0-Ry):min(H-BH,y0+Ry))
        
        candidate=img(l:l+BH-1,k:k+BW-1);
        error=sum(sum(abs(template-candidate)));
        if (error<maxerror)
            xm=k;
            ym=l;
            maxerror=error;
        end
    end
end
        