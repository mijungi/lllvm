function [G, dmat] = makeG(Y,N,dy,nbr_flag)
% Make graph based on 'nbr_flag'
% 'Y' is design matrix. dy x N.
% 'N' Number of datapoints
% 'dy' dimension of each datapoint


dmat = distmat(Y,N,dy); % Distance matrix

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
    %%
    k = nbr_flag;
%     k=3 
    nearness = sort(dmat,1);
    % Wittawat: Should be nearness(k+1, :) ?
    G = (dmat <= repmat(nearness(k,:),N,1)) - eye(N);
    % Wittawat: This + makes it so that G_ij=1 if i is within the kNNs of j and vice versa.
    % So for each row of G, there may be more than K entries of 1.
    G = G + G'; % Makes the k-NN graph symmetric, (forces neighborness)
    G = G >= 1;
%      L = diag(sum(G,2))-G;
%      rank(double(L))
    fprintf(['Using ' num2str(k) '-NN symmetric graph\n']);
else
    % All points are neighbors of the rest
    G = ones(N) - eye(N);
    fprintf(['Using graph, all points are neighbors of each other\n']);
end

end % \MakeG
