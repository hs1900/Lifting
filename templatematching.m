function [dx,dy,matchblock]=templatematching(template,img,x0,y0,R);
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
