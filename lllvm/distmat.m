function dmat = distmat(Y,N,dy)
% Euclidean distance matrix for dataset
% 'Y' is design matrix
% 'N' Number of datapoints
% 'dy' dimension of each datapoint

Y = Y';
numels = N*N*dy;
opt = 2; 
if numels > 5e4 
    opt = 3; 
elseif dy < 20 
    opt = 1; 
end

fprintf(['Calculating graph using opt ' num2str(opt) '\n']);

% distance matrix calculation options
switch opt
    case 1 % half as many computations (symmetric upper triangular property)
        [k,kk] = find(triu(ones(N),1));
        dmat = zeros(N);
        dmat(k+N*(kk-1)) = sqrt(sum((Y(k,:) - Y(kk,:)).^2,2));
        dmat(kk+N*(k-1)) = dmat(k+N*(kk-1));
    case 2 % fully vectorized calculation (very fast for medium inputs)
        a = reshape(Y,1,N,dy); b = reshape(Y,N,1,dy);
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
