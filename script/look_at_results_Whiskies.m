close all;
clear all;

n = 86;
kmat = floor(n./[2, 4, 8, 10, 12, 14, 16, 18, 22, 30, 50]);
seedmat = 1:50;

maxiter = 100;
lower_bound_mat = zeros(maxiter, length(seedmat), length(kmat));

funcs = funcs_global();

load whisky.mat

Gerr = zeros(length(kmat), length(seedmat));

[~, dmat] = makeG(whiskycoords', n,2, k);
[sorted_val, sort_idx] = sort(dmat(72,:)); % the top north one is the center of distances
col = sorted_val;

figure(1)
subplot(221);
scatter( whiskycoords(sort_idx,1),  whiskycoords(sort_idx,2), 40, col, 'o', 'filled');

for k_idx = 1:length(kmat)
    
    k = kmat(k_idx);
    [Gtrue, ~] = makeG(whiskycoords', n, 2, k);
    
    for seed_idx = 1:length(seedmat)
        
        oldRng = rng();
        seed = seedmat(seed_idx);
        rng(seed);
        
        fprintf(['Testing: k= ' num2str(k) ' and seed = ' num2str(seed) '\n'])
        
        % (1) load the file
        filename = ['whiskies_k_' num2str(k) '_s_' num2str(seed)];
        filename_with_directory = funcs.scriptSavedFile([filename '.mat']);
        
        load(filename);
        % this file includes: 'variational_lwb', 'meanCmat', 'meanXmat', 'alphamat', 'betamat', 'gammamat', 'covCmat', 'covXmat'
        lower_bound_mat(:,seed_idx,k_idx) = variational_lwb;
        
        % (2) visualize results:
        %         figure(10*k);
        %         hold on;
        %         subplot(221); plot(variational_lwb);
        %
        %         figure(10*k+1);
        %         clf;
        which_to_show = 10;
        xx = reshape(meanXmat(:,which_to_show), 2, []);
        [Gest, ~] = makeG(xx, n,2, k);
        Gerr(k_idx, seed_idx) = sum(sum((Gtrue-Gest).^2));
        %         subplot(222); scatter(xx(1,:), xx(2,:), 40, col, 'o', 'filled');
        %         title('LL-LVM')
        
        rng(oldRng);
        
    end
    
    
end
%% lowerbound plotting

% mean_lwb = squeeze(mean(lower_bound_mat,2));
% [val, idx] = max(mean_lwb(end,:));
% plot(mean_lwb)
% 
% k = kmat(idx);
seedmat = 0;
k=4;
load whisky.mat

% define essential quantities
Y =  whiskydata' ; 
dy = size(Y,1);
n = size(Y,2);

figure(1)
subplot(231);
[~, dmat] = makeG(whiskycoords', n,2, k);
[sorted_val, sort_idx] = sort(dmat(72,:)); % the top north one is the center of distances
col = sorted_val;
scatter( whiskycoords(sort_idx,1),  whiskycoords(sort_idx,2), 40, col, 'o', 'filled');
[Gtrue, ~] = makeG(whiskycoords', n,2, k);
subplot(234);
% imagesc(Gtrue);title('G true');
gplot(Gtrue, whiskycoords, '*-');

[mappedX] = gplvm(Y', 2);
subplot(233); scatter(mappedX(sort_idx,1), mappedX(sort_idx,2), 40, col, 'o', 'filled');
title('GPLVM');
[Gest_gplvm, ~] = makeG(mappedX', n,2, k);
subplot(236);
% imagesc(Gest_gplvm); 
gplot(Gest_gplvm, whiskycoords, '*-'); title('G from gplvm');

for seed_idx = 1:length(seedmat)
    
    oldRng = rng();
    seed = seedmat(seed_idx);
    rng(seed);
    
    fprintf(['Testing: k= ' num2str(k) ' and seed = ' num2str(seed) '\n'])
    
    % (1) load the file
    funcs = funcs_global(); 
    filename = ['whiskies_k_' num2str(k) '_s_' num2str(seed)];
    filename_with_directory = funcs.scriptSavedFile([filename '.mat']);
    
    load(filename_with_directory);
%     lower_bound_mat(:,seed_idx,k_idx) = variational_lwb;
    
    % (2) visualize results:
    which_to_show = 10;
    xx = reshape(meanXmat(:,which_to_show), 2, []);
    subplot(232); scatter(xx(1,sort_idx), xx(2,sort_idx), 40, col, 'o', 'filled');
    title('LL-LVM')
    
    [Gest, ~] = makeG(xx, n,2, k);
    subplot(235);
    gplot(Gest, whiskycoords, '*-'); title('G from lllvm');
%     imagesc(Gest); title('G from lllvm');
    
    
    [sum(sum((Gtrue-Gest).^2)) sum(sum((Gtrue-Gest_gplvm).^2))]
    
%     pause;
    
    rng(oldRng);
    
end
% 
% [mappedX, mapping] = lle(Y', 2, k);
% subplot(222); scatter(mappedX(:,1), mappedX(:,2), 40, col, 'o', 'filled');
% title('LLE');

% k = n./[2, 4, 8];
% hold on;
% shadedErrorBar(1:maxiter, mean(lower_bound_mat(:,:,2),2), std(lower_bound_mat(:,:,2),0, 2), 'b')
% shadedErrorBar(1:maxiter, mean(lower_bound_mat(:,:,3),2), std(lower_bound_mat(:,:,3),0, 2), 'r')
% shadedErrorBar(1:maxiter, mean(lower_bound_mat(:,:,4),2), std(lower_bound_mat(:,:,4),0, 2), 'g')

