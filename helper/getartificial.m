function [N, n, Y, G, dmat, col, X] = getartificial(N_in, data_flag, nbr_flag)

% this is what Ahmad wrote
% 'N_in'        : N data points
% 'data_flag'   : flag for data_handle function
%                     1 : 3D Gaussian
%                     2 : Punctured Sphere by Saul & Roweis
%                     3 : Swiss Roll
%                     4 : Twin Peaks by Saul & Roweis (WJ: broken)
%                     5 : Cornered Plane (WJ: broken)
% 'opts'        : [latent dim, neighborhood selection method flag]
%                 [m         , nbr_flag                          ]
% 'nbr_flag'    : As follows
%                   <1,	All points are neighbors of the rest
%                   1,	Guarantees every point at least one neighbor
%                   2,  Uses mean distance as threshold
%                   3,  Uses temporal neighborhoods (2 time-steps) FOR
%                       NEURAL DATA ONLY
%                   >3,	Uses k-nearest neighbors (where k = nbr_flag)
% 'N_itr'       : How many iterations of learning do you want?


if N_in <= 0
    fprintf(['The value of N, ' num2str(N_in) ...
        ', is not allowed for this data. Using N = 100.\n']);
    N = 100;
else
    N = N_in;
end

n = 3; % change this if necessary when introducing new data

param = 1; %could be STD, or something else

fprintf(['Getting artificial ' num2str(N) ' data points: ']);

if data_flag == 1 % 3D Gaussian
    fprintf('3D Gaussian \n');
    Y = param * randn(N,n);
    X = Y(:,1:2)';
    Y(:,n) = 1 / (param^2 * 2 * pi) * exp ( (-Y(:,1).^2 - Y(:,2).^2) / (2*param^2) );
    Y = Y';
elseif data_flag ==2
    fprintf('Swiss Roll with a hole \n');
    % Swiss Roll w/ hole example taken from Donoho & Grimes
    tt = (2.5*pi/2)*(1+2*rand(1,2*N));
    height = 10*rand(1,2*N);
    kl = repmat(0,1,2*N);
    for ii = 1:2*N
        if ( (tt(ii) > 9)&(tt(ii) < 12))
            if ((height(ii) > 9) & (height(ii) <14))
                kl(ii) = 1;
            end;
        end;
    end;
    kkz = find(kl==0);
    tt = tt(kkz(1:N));
    height = height(kkz(1:N));
    Y = [tt.*cos(tt); height; param*tt.*sin(tt)];
    col = tt; 
    X = [tt; height]; 
    % elseif data_flag == 2 % Punctured Sphere by Saul & Roweis
    %     fprintf('Punctured Sphere by Saul & Roweis \n');
    %     inc = 9/sqrt(N);   %inc = 1/4;
    %     [xx,yy] = meshgrid(-5:inc:5);
    %     rr2 = xx(:).^2 + yy(:).^2;
    %     [~, ii] = sort(rr2);
    %     X = [xx(ii(1:N))'; yy(ii(1:N))'];
    %     a = 4./(4+sum(X.^2));
    %     Y = [a.*X(1,:); a.*X(2,:); param*2*(1-a)];
elseif data_flag == 3   % Swiss Roll
    fprintf('Swiss Roll \n');
    tt = (2.5*pi/2)*(1+2*rand(1,N));
    height = 10*rand(1,N);
    Y = [0.35*tt.*cos(tt); height; 0.35*param*tt.*sin(tt)];
    X =  [tt; height];
elseif data_flag == 4   % Twin Peaks by Saul & Roweis
    fprintf('Twin Peaks by Saul & Roweis \n');
    inc = 1.5 / sqrt(N);  % inc = 0.1;
    [xx2,yy2] = meshgrid(-1:inc:1);
    zz2 = sin(pi*xx2).*tanh(3*yy2);
    xy = 1-2*rand(2,N);
    Y = [xy; sin(pi*xy(1,:)).*tanh(3*xy(2,:))];
elseif data_flag == 5 % Cornered Plane
    fprintf('Cornered Plane \n');
    k = 1;
    xMax = floor(sqrt(N));
    yMax = ceil(N/xMax);
    cornerPoint = floor(yMax/2);
    for x = 0:xMax
        for y = 0:yMax
            if y <= cornerPoint
                Y(k,:) = [x,y,0];
                tt(k) = y;
            else
                Y(k,:) = [x,cornerPoint+(y-cornerPoint)*cos(pi*param/180),(y-cornerPoint)*sin(pi*param/180)];
                tt(k) = y;
            end
            k = k+1;
        end
    end
    tt = tt(1:N)';
    Y = Y(1:N,:)';
elseif data_flag == 6 % Generatively build data
    fprintf('Generatively build ... ');
    fname = vardimred(N, @getartificial, 1, [2, 2]);
    fprintf(['using file ' fname ' \n']);
    raw = load([fname '.mat']);
    eval(['data = raw.learning' num2str(raw.itr) ';']); % Get latest iter data
    Y = reconstructY(data.X_Mu,data.C_Mu,raw.fixedp.N,raw.fixedp.m,raw.fixedp.n);
    G = raw.fixedp.G;
    dmat = raw.fixedp.dmat;
else
    error('Invalid data flag');
end

% At this point Y is a design matrix
%Y = Y ./ max(sum(Y.*Y));          % normalize
%Y = Y-repmat(mean(Y,2),1,N);      % center
if data_flag ~= 6 % Because we outupt generative G for that
    [G, dmat] = makeG(Y,N,n,nbr_flag);
    %     [G, dmat] = makeG(X,N,n,nbr_flag);
end
Y = reshape(Y,n*N,1);               % vectorize

% Set colors (classes, or regressions)
if (data_flag == 3) || (data_flag == 5)
    col = tt;
else
    plotY = reshape(Y,n,N);
    col   = plotY(3,:);
end

end % \getartificial


%%
function [G, dmat] = makeG(Y,N,n,nbr_flag)
% Make graph based on 'nbr_flag'
% 'Y' is design matrix
% 'N' Number of datapoints
% 'n' dimension of each datapoint

dmat = distmat(Y,N,n); % Distance matrix

if nbr_flag == 1
    % Guarantees everyone at least one neighbor
    threshold = dmat;
    threshold(threshold == 0) = Inf;
    threshold = max(min(threshold));
    G = (dmat <= threshold*1.01 ) - eye(N); % Graph
    fprintf(['Using graph threshold of ' num2str(threshold) ' that guarantees everyone at least one neighbor\n']);
elseif nbr_flag == 2
    % Uses mean distance as threshold
    % figure(2);
    % hist(reshape(dmat,numel(dmat),1)); title('Histogram of distances');
    threshold = mean(reshape(dmat,numel(dmat),1));
    G = (dmat <= threshold*1.01 ) - eye(N); % Graph
    fprintf(['Using mean graph threshold of ' num2str(threshold) '\n']);
elseif nbr_flag >= 3
    % Uses k-nearest neighbors
    k = nbr_flag;
    nearness = sort(dmat,1);
    G = (dmat <= repmat(nearness(k,:),N,1)) - eye(N);
    G = G + G'; % Makes the k-NN graph symmetric, (forces neighborness)
    G = G >= 1;
    fprintf(['Using ' num2str(k) '-NN symmetric graph\n']);
else
    % All points are neighbors of the rest
    G = ones(N) - eye(N);
    fprintf(['Using graph, all points are neighbors of each other\n']);
end

end % \MakeG


function dmat = distmat(Y,N,n)
% Distance matrix for dataset
% 'Y' is design matrix
% 'N' Number of datapoints
% 'n' dimension of each datapoint

Y = Y';
numels = N*N*n;
opt = 2; if numels > 5e4, opt = 3; elseif n < 20, opt = 1; end

fprintf(['Calculating graph using opt ' num2str(opt) '\n']);

% distance matrix calculation options
switch opt
    case 1 % half as many computations (symmetric upper triangular property)
        [k,kk] = find(triu(ones(N),1));
        dmat = zeros(N);
        dmat(k+N*(kk-1)) = sqrt(sum((Y(k,:) - Y(kk,:)).^2,2));
        dmat(kk+N*(k-1)) = dmat(k+N*(kk-1));
    case 2 % fully vectorized calculation (very fast for medium inputs)
        a = reshape(Y,1,N,n); b = reshape(Y,N,1,n);
        dmat = sqrt(sum((a(ones(N,1),:,:) - b(:,ones(N,1),:)).^2,3));
    case 3 % partially vectorized (smaller memory requirement for large inputs)
        dmat = zeros(N,N);
        for k = 1:N
            dmat(k,:) = sqrt(sum((Y(k*ones(N,1),:) - Y).^2,2));
        end
    case 4 % another compact method, generally slower than the others
        a = (1:N);
        b = a(ones(N,1),:);
        dmat = reshape(sqrt(sum((Y(b,:) - Y(b',:)).^2,2)),N,N);
end

end % \distmat
