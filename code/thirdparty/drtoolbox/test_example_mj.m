%% test examples

clear all;
close all;
clc;

addpath techniques

% swiss roll
n = 2000; 
[Y, ~, t] = generate_data('swiss', n);
no_dims = round(intrinsic_dim(Y, 'MLE'));
% [X, mapping] = compute_mapping(Y, 'Isomap', no_dims);
% scatter(X(:,1), X(:,2)); title('isomap');
k = 6; 
[X, mapping] = compute_mapping(Y, 'LLE', no_dims, k);
% figure, scatter(X(:,1), X(:,2)); title('LLE');

length(X)
%%
[sortedX, idx_sort] = sortrows(X);
% 
% [1 0 0]
% [0 1 0]
% [0 0 1]
n = length(sortedX);

color_x = linspace(0, 1, n);
color_x = fliplr(color_x);
color_y = linspace(0, 1, n/2);
color_y = [color_y fliplr(color_y)];
color_z = linspace(0, 1, n);
Color = [color_x' color_y' color_z']; 

subplot(121); scatter(sortedX(:,1), sortedX(:,2), 30,  Color); 
% title('isomap')
% , 'markerfacecolor', Color)

sortedY = Y(idx_sort, :);
subplot(122); scatter3(sortedY(:,1), sortedY(:,2), sortedY(:,3), 30, Color); title('swiss roll');

%%

% howmanyneighbors = 20; 
% [IDX,D] = knnsearch(Y,Y,'K',howmanyneighbors);
% nk = n/howmanyneighbors; 
% 
% 
% 
% for i=1:n
%     withinneighbor = Y(IDX(i,:), :); 
% 





%%

figure, scatter3(X(:,1), X(:,2), X(:,3), 5, labels); title('Original dataset'), drawnow
no_dims = round(intrinsic_dim(X, 'MLE'));
disp(['MLE estimate of intrinsic dimensionality: ' num2str(no_dims)]);
[mappedX, mapping] = compute_mapping(X, 'LLE', no_dims, 20);
figure, scatter(mappedX(:,1), mappedX(:,2), 30, Color); title('LLE');

[mappedX, mapping] = compute_mapping(X, 'Isomap', no_dims);
figure, scatter(mappedX(:,1), mappedX(:,2), 5, labels); title('isomap');

    
    %%     
    [mappedX, mapping] = compute_mapping(X, 'Laplacian', no_dims, 7);	
	figure, scatter(mappedX(:,1), mappedX(:,2), 5, labels(mapping.conn_comp)); title('Result of Laplacian Eigenmaps'); drawnow



