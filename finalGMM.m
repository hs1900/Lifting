%% LIFTING
clear;clc;close all;
GMM = 'GMM.avi';
Obj = VideoWriter(GMM);
writerObj.FrameRate = 30;
open(Obj);

vidIn = 'badformshade.m4v';
vidObj = VideoReader(vidIn); 
nFrames = vidObj.NumberOfFrames;
vidHeight = vidObj.Height; 
vidWidth = vidObj.Width;

% Train frames 
foregroundDetector = vision.ForegroundDetector('NumGaussians', 3,...
    'NumTrainingFrames', 130,'LearningRate',.0001);

% Complete background subtraction and Write Video
videoReader = vision.VideoFileReader(vidIn);

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
% If there are multiple bounding boxes
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
begcutoff= floor(nFrames*(1/4)); %processing only the first-second third of the video, this number can be changed
endcutoff = floor(nFrames*(3/4));

bendingBBox = BBox(begcutoff:endcutoff,3); %locating the bbox with largest width
standingBBox = BBox(begcutoff:endcutoff,4); % locating the bbox with largest height

tempmaxH = find(standingBBox == max(standingBBox),1);
standingFrameLoc = tempmaxH + begcutoff -1; %frame location where person is standing

 % Locate Butt
tempmaxW = find(bendingBBox == max(bendingBBox),1);
bendingFrameLoc = tempmaxW+begcutoff-1; %frame location where person is bending
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
for m = bendingFrameLoc-1:-1:begcutoff
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
       
for m = bendingFrameLoc+1:endcutoff
    nextFrame = rgb2gray(read(vidObj,m));
    [dx, dy, matchblock] = templatematching(backTemp,nextFrame,buttCol-buttTempColStart,buttRow-buttTempRowStart ,1);

    imshow(uint8(matchblock));
    backTemp = matchblock;
    
    buttCol = buttCol+dx;
    buttRow = buttRow+dy;
    
   buttCoordsTemp(m,1) = buttCol-buttTempColStart;
   buttCoordsTemp(m,2) = buttRow-buttTempRowStart;
end

%% Pinpointing lowerback as point of interest

%after obtaining the locations for the back template, we find the rightmost point in each frame
buttObj = VideoReader(GMM);
buttPOI = zeros(nFrames,2);

for m = begcutoff:endcutoff
    frame = read(buttObj,m);
    bFrame = frame(buttCoordsTemp(m,2):buttTempRowStart+buttCoordsTemp(m,2),buttCoordsTemp(m,1):buttTempColStart+buttCoordsTemp(m,1)+buttTempColEnd);
%     imshow(bFrame);
    binaryMatrix = bFrame > 200;
    
    [sel c] = max(binaryMatrix ~= 0, [], 1);
    binLoc = sel.*c;
    binCol = max(find(binLoc));
    binRow = min(find(binaryMatrix(:,binCol)));
    
    buttPOI(m,1) = binCol+buttCoordsTemp(m,1);
    buttPOI(m,2) = binRow+buttCoordsTemp(m,2);
end
%% Find standing frame

% Set Range and Analyze

tempboxH = BBox(begcutoff:endcutoff,4);

% Locate Neck
tempmax = find(tempboxH == max(tempboxH),1);
standingFrameLoc = tempmax+begcutoff-1;
vidObj2 = VideoReader(GMM);

standingFrameLoc = standingFrameLoc(1,:);
EndFrame = (read(vidObj2,standingFrameLoc));

%% Testing Curvatures
clc;close all;
GMMObj = VideoReader(GMM); 
fontSize = 20;

% Use Face to Obtain Frame 
testFrame = rgb2gray(read(GMMObj,standingFrameLoc));
bwFrame = bwboundaries(testFrame);

BW = testFrame > 128;
[B,L] = bwboundaries(BW,'noholes');
figure;
imshow(label2rgb(L, @jet, [.5 .5 .5]))
hold on;
%initialize size
Bcell = size(B{1},1);
k=1;
for i = 2:size(B,1)
    if size(B{i},1) > Bcell
        k = i;
    end
end
% for k = 1:length(B)
   boundary = B{k};
   plot(boundary(1:floor(length(boundary)),2), ...
   boundary(1:floor(length(boundary)),1), 'r', 'LineWidth', 2)
%end

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

	curvature(t) = 2*abs((x2-x1).*(y3-y1)-(x3-x1).*(y2-y1)) ./ ...
	sqrt(((x2-x1).^2+(y2-y1).^2)*((x3-x1).^2+(y3-y1).^2)*((x3-x2).^2+(y3-y2).^2));
end


threshold = 1;
cur = curvature > threshold;
Candidates = cur .*curvature;
% figure;plot(Candidates);


indexCurv = find(Candidates);
pointCand = boundary(indexCurv,:);
plot(pointCand(:,2),pointCand(:,1),'x','MarkerSize' ,20);
% 2 = x, 1 = y
% Plot curvature.
figure;plot(curvature, 'b-', 'LineWidth', 1)
grid on;
xlim([1 numberOfPoints]); % Set limits for the x axis.
title('Radius of Curvature', 'FontSize', fontSize);

%% HOG 
% vidIn = 'badformshade.m4v';
% vidObj = VideoReader(vidIn); 
hogFrame = rgb2gray(read(vidObj,standingFrameLoc));
figure;
imshow(hogFrame);
% [featureVector,hogVisualization] = extractHOGFeatures(hogFrame);
% hold on;
% plot(hogVisualization);
% nBlock = length(featureVector)/324;

%Template neck
% I = rgb2gray(read(vidObj,650));
% hogtemp = I(44-16:44+16,316-16:316+16);
% [tempFeature,tempVisualization] = extractHOGFeatures(hogtemp);
%tempFeature = [ 0.072607093	0.13456741	0.37930390	0.28385928	0.12277971	0.026224004	0.011287540	0.017976983	0.030027129	0.044516113	0.12246862	0.17392057	0.13812861	0.065385096	0.022635803	0.018405246	0.025685877	0.016091088	0.090995789	0.37930390	0.37930390	0.18034941	0.027008183	0.0098466119	0.0078076376	0.0066125123	0.0081123216	0.14518782	0.37930390	0.37930390	0.13069528	0.054355603	0.025455050	0.019135546	0.030762224	0.048367094	0.13873145	0.27921543	0.23108560	0.15861881	0.15168695	0.067026839	0.051243655	0.063814066	0.063953944	0.082654141	0.060229201	0.055715654	0.055773996	0.15188819	0.071211860	0.039646070	0.055929180	0.093825608	0.27921543	0.27921543	0.27921543	0.24020818	0.18792391	0.071848094	0.054842547	0.091033958	0.19289106	0.24466138	0.27921543	0.26272869	0.14459625	0.23223160	0.11164730	0.074232191	0.077458598	0.20346566	0.12396111	0.073518232	0.077454381	0.055010572	0.18559486	0.081365608	0.045779746	0.065744683	0.13907705	0.24907146	0.13826142	0.065701470	0.021902164	0.073196582	0.044532649	0.090713196	0.10084448	0.25010991	0.25010991	0.25010991	0.25010991	0.16908018	0.25010991	0.17209801	0.13429125	0.12334946	0.25010991	0.25010991	0.25010991	0.17883375	0.10467339	0.11639597	0.096881993	0.11903451	0.23915868	0.25010991	0.072682768	0.35613865	0.35613865	0.085364260	0.011528507	0.0059318771	0.0063078455	0.0074027400	0.0090820044	0.27309611	0.35613865	0.33708191	0.065519698	0.023081252	0.012294510	0.012076160	0.020826690	0.068314627	0.056665946	0.31108508	0.16896193	0.010789772	0.028975517	0.0040534306	0.0034424092	0.0056581139	0.014600356	0.35613865	0.35613865	0.12695260	0.019150108	0.012004781	0.010444565	0.0057357075	0.0095224557	0.088651262	0.32208478	0.34876701	0.19629472	0.050244417	0.036297604	0.013854353	0.013211690	0.021868335	0.091952264	0.22185668	0.20360321	0.075396620	0.038297426	0.052116886	0.030597178	0.020351529	0.030054195	0.073857971	0.34876701	0.34876701	0.099179886	0.061611034	0.029857054	0.056507159	0.0080305925	0.031545613	0.11449248	0.34876701	0.34876701	0.12268881	0.16572593	0.10798324	0.11809546	0.0093231481	0.11532797	0.12143803	0.24710435	0.18358113	0.084808990	0.043435901	0.069136441	0.044985179	0.035873193	0.050850663	0.089822359	0.13459115	0.092351653	0.051738370	0.028140502	0.027382944	0.027597131	0.035487678	0.071863316	0.13114330	0.37414250	0.37414250	0.16315362	0.18772754	0.13056695	0.12725621	0.016151166	0.15237014	0.18521710	0.37414250	0.37414250	0.064180456	0.014916847	0.030656183	0.030380564	0.027611233	0.061357163	0.31403807	0.12520045	0.35540456	0.11294682	0.035206545	0.14333041	0.012632977	0.016413527	0.018020667	0.047205966	0.54322290	0.54322290	0.19861075	0.064336151	0.086998940	0.057732411	0.041235607	0.034315832	0.29228321	0.0072003412	0.018034542	0.039471470	0.011015114	0.20033348	0.00070607965	0.010684592	0.0059004263	0.0069564995	0.029379519	0.046237450	0.032854144	0.023988971	0.18923137	0.028121067	0.040426064	0.037483186	0.028816022	0.32471621	0.32471621	0.078424975	0.073503129	0.086201958	0.11349218	0.013117575	0.036314901	0.10225528	0.32471621	0.32471621	0.13978489	0.21784361	0.24464221	0.23662008	0.012663602	0.12553538	0.11878593	0.010562550	0.014542010	0.015463577	0.021345077	0.27109978	0.15098867	0.014265817	0.014195021	0.010611554	0.031955719	0.041689567	0.039859153	0.082498290	0.32471621	0.32471621	0.026769148	0.023948018	0.020204458	0.34049854	0.34049854	0.13925594	0.18716404	0.20534830	0.18414250	0.013420601	0.11699987	0.13979729	0.34049854	0.34049854	0.048043210	0.017472474	0.036373995	0.026934303	0.015404203	0.039088346	0.27743644	0.065618917	0.071145572	0.044065826	0.080153480	0.34049854	0.26601973	0.027899725	0.026298145	0.026932299	0.24453455	0.092656814	0.035943888	0.032174625	0.065808438	0.034161109	0.0097178910	0.019976631	0.074814901];
tempFeature = [0.14943308,0.36169705,0.25075939,0.10627335,0.038552579,0.020293027,0.017680168,0.031107783,0.045924269,0.10839719,0.10950667,0.073320597,0.038361929,0.057253391,0.028863683,0.015057934,0.029198637,0.076166198,0.38888010,0.38888010,0.17983225,0.031269766,0.011060436,0.010597482,0.0068135709,0.024044484,0.13070711,0.38888010,0.38888010,0.15366217,0.11201661,0.037162151,0.048065886,0.012115389,0.070827842,0.15689355,0.14910607,0.12638235,0.084377281,0.048281077,0.089181729,0.052441228,0.038342498,0.044972960,0.10854328,0.14321536,0.082613200,0.041862521,0.023397146,0.057893973,0.037654247,0.058324602,0.047412764,0.12011989,0.39485797,0.39485797,0.23642258,0.17926225,0.073651075,0.081411608,0.031588700,0.13307749,0.23843592,0.39485797,0.39485797,0.074165240,0.050682995,0.049238179,0.032636046,0.045286119,0.075887248,0.23069611,0.20585197,0.096935771,0.047524355,0.019010043,0.051262766,0.027855435,0.063082740,0.094617642,0.21524976,0.31529686,0.15799221,0.094012402,0.15282197,0.11081563,0.036965009,0.050108742,0.23492187,0.31529686,0.31529686,0.31529686,0.071127214,0.056293570,0.056044966,0.036008190,0.049014103,0.094371997,0.31386727,0.31529686,0.12991585,0.030254848,0.029318158,0.035860043,0.032758929,0.031619947,0.11605231,0.31529686,0.36806637,0.36806637,0.10393218,0.022060679,0.016141836,0.016146349,0.0049375664,0.018425619,0.11584447,0.36806637,0.36806637,0.14194837,0.15852198,0.11577772,0.13103870,0.012846406,0.084531397,0.13936672,0.035547670,0.036354057,0.015114535,0.025208611,0.080722235,0.039258882,0.013552477,0.013123359,0.023511097,0.10641160,0.092846639,0.064116947,0.13741675,0.36806637,0.36806637,0.015502543,0.045484047,0.036307108,0.31926590,0.31926590,0.14909510,0.16354160,0.12500131,0.12777841,0.013808588,0.096546806,0.14272234,0.31926590,0.31926590,0.054261211,0.041427381,0.030221775,0.015753925,0.013934117,0.047114827,0.19007556,0.15505035,0.12753296,0.072216317,0.14647682,0.31926590,0.31926590,0.016392559,0.049159531,0.046410769,0.31926590,0.18413204,0.048270155,0.058219813,0.074987188,0.034978881,0.012720381,0.026296061,0.11032271,0.35365033,0.32208440,0.047489420,0.040823381,0.028651781,0.019586258,0.016452348,0.053271711,0.26746410,0.35365033,0.10534272,0.016718592,0.017532174,0.019742755,0.018552095,0.018553814,0.064658739,0.33028439,0.35365033,0.19638878,0.048819929,0.055787046,0.052172348,0.028992325,0.018042944,0.029583959,0.17179762,0.35365033,0.075542472,0.011751402,0.017403416,0.024821388,0.018260770,0.025946543,0.029918479,0.32121152,0.020588618,0.022616778,0.030839132,0.040676985,0.22837965,0.083177097,0.033405311,0.022887878,0.029162398,0.019415017,0.018181225,0.067157619,0.15243405,0.44996640,0.44996640,0.022724397,0.029500235,0.045949489,0.025684932,0.022678502,0.037644044,0.024567412,0.29254043,0.064590171,0.027854085,0.017048258,0.026500123,0.028675457,0.012724980,0.021869488,0.046659309,0.44996640,0.44226548,0.018247571,0.019989045,0.029074052,0.033297684,0.032138988,0.079422146,0.17833909,0.45059204,0.45059204,0.027751902,0.037282124,0.057303071,0.14285722,0.080769897,0.090244830,0.11242373,0.17482086,0.089956686,0.033877548,0.050902959,0.11090814,0.036577009,0.017142270,0.025196115,0.046669576,0.45059204,0.45059204,0.026530119,0.030818144,0.038319372,0.061776359,0.063190952,0.061805036,0.039600924,0.090741813,0.054123912,0.038242910,0.066953801,0.043433707,0.37359884,0.17732963,0.17195454,0.19704777,0.23544154,0.13622662,0.081974372,0.11443925,0.30518234,0.37359884,0.12787466,0.048876163,0.063260853,0.12958607,0.093899794,0.10971537,0.079432182,0.37359884,0.11363317,0.13596140,0.13489829,0.098220728,0.17542428,0.11172237,0.078986697,0.13289388,0.071768545,0.11223330,0.051333051,0.058009341,0.097753510,0.20012201,0.12205683,0.055816278,0.038755201,0.092313975];
% %Template shoulder
% I = rgb2gray(read(vidObj,650));
% hogtemp = I(52:84,304:336);
% figure;
% [tempFeature,tempVisualization] = extractHOGFeatures(hogtemp);

% tempFeature = reshape(tempFeature,36,9);
% imshow(hogtemp);
% hold on;
% plot(tempVisualization);
% ihogCand = 99999999;
% for i = 20000:324:length(featureVector)-324
%     
%     hogCand = (sum(abs(featureVector(i:i+324-1)-tempFeature)));
%     if hogCand<ihogCand
%         ihogCand = hogCand;
%         hogLoc = i;
%     end
% end
ihogCand = 99999;
for i = 1:size(pointCand,1)
    if pointCand(i,1)>16 && pointCand(i,2)>16
    testTemp = hogFrame(pointCand(i,1)-16:pointCand(i,1)+16,...
        pointCand(i,2)-16:pointCand(i,2)+16);
    testFeature = extractHOGFeatures(testTemp);
    hogCand = mean(abs(testFeature - tempFeature));
    
    if hogCand < ihogCand
        ihogCand = hogCand;
        hogLoc = i;
    end
    end
end

testTemp = hogFrame(pointCand(hogLoc,1)-16:pointCand(hogLoc,1)+16,...
        pointCand(hogLoc,2)-16:pointCand(hogLoc,2)+16);
    
imshow(testTemp);
[testFeature,testVisualization] = extractHOGFeatures(testTemp);hold on
plot(testVisualization);

%% template match around Neck area

neckRow = pointCand(hogLoc,1);
neckCol = pointCand(hogLoc,2);
neckTemp = hogFrame(neckRow-20:neckRow+60,neckCol-30:neckCol+20);

neckCoords = zeros(nFrames,2);
neckCoords(standingFrameLoc,1) = pointCand(hogLoc,1);
neckCoords(standingFrameLoc,2) = pointCand(hogLoc,2);

figure;
%matching backwards
for m = standingFrameLoc-1:-1:begcutoff
    prevFrame = rgb2gray(read(vidObj,m));
    [dx, dy, matchblock] = templatematching(neckTemp,prevFrame,neckCol-30,neckRow-20,1);
    
    neckTemp = matchblock;
    imshow(uint8(matchblock));
    neckRow = neckRow + dy;
    neckCol = neckCol + dx;
    
    neckCoords(m,1) = neckRow - 20;
    neckCoords(m,2) = neckCol - 30;
    
end
%% matching forwards

neckRow = pointCand(hogLoc,1);
neckCol = pointCand(hogLoc,2);
neckTemp = hogFrame(neckRow-20:neckRow+60,neckCol-30:neckCol+20);

for m = standingFrameLoc+1:endcutoff
    nextFrame = rgb2gray(read(vidObj,m));
    [dx, dy, matchblock] = templatematching(neckTemp,nextFrame,neckCol-30,neckRow-20,1);

    neckTemp = matchblock;
    imshow(uint8(matchblock));
    
    neckRow = neckRow + dy;
    neckCol = neckCol + dx;
    
    neckCoords(m,1) = neckRow - 20;
    neckCoords(m,2) = neckCol - 30;
    
end

%% Pinpointing upperback as point of interest

buttObj = VideoReader(GMM);
backPOI = zeros(nFrames,2);

for m = begcutoff:endcutoff
    frame = read(buttObj,m);
    bFrame = frame(neckCoords(m,1):80+neckCoords(m,1),neckCoords(m,2):50+neckCoords(m,2));
    binaryMatrix = bFrame > 200;
    imshow(bFrame)
    if sum(binaryMatrix(1,:))~=0
        binCol = max(find(binaryMatrix(1,:)));
        binRow = 0;
    else
        binCol = 0;
        binRow = min(find(binaryMatrix(:,1)));
    end
    
    
    backPOI(m,1) = binRow+neckCoords(m,1);
    backPOI(m,2) = binCol+neckCoords(m,2);
end
%% Plotting Points

for i = 1:nFrames
    finalFrame = read(vidObj,i) ;
    imshow(finalFrame); hold on
    
    plot(backPOI(i,2),backPOI(i,1),'o','MarkerSize',20);
    plot(buttPOI(i,1),buttPOI(i,2),'o','MarkerSize',20); 
end

%%
%Template shoulder
 I = rgb2gray(read(vidObj,standingFrameLoc));
 hogtemp = I(52:84,304:336);
 figure;
 imshow(hogtemp);hold on
 [tempFeature,tempVisualization] = extractHOGFeatures(hogtemp);
 plot(tempVisualization);
