% to make a figure about simulated data drawn from the 3D Gaussian
% mijung wrote on Oct 28, 2015

clear;
clc;

% data_flag = 1;
% n  = 400;
% dy = 3;
% dx = 2;
% 
% kmat = 4:2:18;

data_flag = 3;
n  = 400;
dy = 3;
dx = 2;

kmat =  6:11;

seedmat = 1:10;

maxiter = 50;
lower_bound_mat = zeros(maxiter, length(seedmat), length(kmat));

for k_idx = 1:length(kmat)
    
    k = kmat(k_idx);
    
    for seed_idx = 1:length(seedmat)
        
        oldRng = rng();
        seed = seedmat(seed_idx);
        rng(seed);
        
        fprintf(['Testing: k= ' num2str(k) ' and seed = ' num2str(seed) '\n'])
        
        %% for color coding, check how I indexed datapoints:
        
        % (1) load the file:
        load(strcat('fixingalpha_dataflag ', num2str(data_flag), 'seednum ', num2str(seed), 'k ', num2str(k), '.mat'));
        
        lower_bound_mat(1:length(results.lwbs),seed_idx,k_idx) = results.lwbs;
        
        %%
        %         % (2) visualize results:
        
%         figure(2);
%         subplot(221); plot(results.lwbs);
%         which_to_show = length(results.lwbs);
%         %         which_to_show = 40;
%         xx = reshape(results.mean_x(:,which_to_show), 2, []);
%         
%         subplot(222); scatter(xx(1,:), xx(2,:), 20, col, 'o', 'filled');
%         title('LL-LVM')
%         
%         figure(1);
%         plotlearning(dx,dy,n,reshape(results.mean_c(:,which_to_show),dy,n*dx),Yraw);
%         
%         
%         pause;
        
        %% wonder how LLE works with the same data
        
        %         load Y_USPS_400;
        %         Y = Y_USPS_400;
        %
        %         %% LLE
        %
        %         if seed_idx==length(seedmat)
        %
        %             dx = 2;
        %
        %             %% LLE
        %
        %             [mappedX, mapping] = lle(Y', dx, k);
        %             subplot(224); scatter(mappedX(:,1), mappedX(:,2), 20, permuted_val, 'o', 'filled');
        %             title('LLE');
        %
        %             %% GP-LVM
        %
        %             mappedX = gplvm(Y', dx);
        %             subplot(223); scatter(mappedX(:,1), mappedX(:,2), 20, permuted_val, 'o', 'filled');
        %             title('GPLVM');
        %
        %             %% Isomap
        %
        %             [mappedX, mapping] = isomap(Y', dx, k);
        %             subplot(222); scatter(mappedX(:,1), mappedX(:,2), 20, permuted_val, 'o', 'filled');
        %             title('Isomap');
        %
        %         end
        %
        %         pause;
        
        rng(oldRng);
        
    end
    
    
end

%% lowerbound plotting

% k = n./[2, 4, 8];
% kmat = floor(n./[8, 6, 4, 3, 2, 1]);
hold on;
seg = 1/length(kmat);

highest_lwb = zeros(length(kmat), length(seedmat));
for i=1:length(kmat)
    
    for j = 1: length(seedmat)
        
        highest_lwb(i, j) = max(lower_bound_mat(:,j,i));
    
    end

end

% figure(2); subplot(223);
plot(kmat, mean(highest_lwb,2), '-o'); set(gca, 'xlim', [5.8 11.2])
% errorbar(kmat, mean(highest_lwb,2),std(highest_lwb,0,2))

%% plotting with k set by the highest lwb

% (1) load the file:
mean_max_lwb = mean(highest_lwb,2); 
[val, idx] = max(mean_max_lwb);
k = kmat(idx);
seed =4;

load(strcat('fixingalpha_dataflag ', num2str(data_flag), 'seednum ', num2str(seed), 'k ', num2str(k), '.mat'));

% (2) visualize results:

figure(2);
% subplot(221); plot(results.lwbs);
which_to_show = length(results.lwbs);
% which_to_show =4;
xx = reshape(results.mean_x(:,which_to_show), 2, []);

scatter(xx(1,:), xx(2,:), 20, col, 'o', 'filled');
title('LL-LVM')

figure(1);
plotlearning(dx,dy,n,reshape(results.mean_c(:,which_to_show),dy,n*dx),Yraw, col);


%% data

% 3D
plotY = reshape(Yraw,dy,n);
scatter3( plotY(1,:) , plotY(2,:) , plotY(3,:) , [] , col , 'o', 'filled');
grid off;

% 2D
scatter(truex(1,:), truex(2,:), 20, col, 'o', 'filled');

%%
% graph
scatter(truex(1,:), truex(2,:), 80, col, 'o', 'filled');
hold on; 
gplot(G,truex','b') 

%%

% for i=1:length(kmat)
%     
%     firstk = lower_bound_mat(:,:,i);
%     % firstk = lower_bound_mat(:,:,1)./kmat(1);
%     firstk(firstk==0) = nan;
%     firstk_mean = nanmean(firstk,2);
%     firstk_std = nanstd(firstk, 0, 2);
% 
%     if i==1
%         plot(1:length(firstk_mean), firstk_mean, 'g');
%     elseif i==2
%         plot(1:length(firstk_mean), firstk_mean, 'r');
%     elseif i==3
%         plot(1:length(firstk_mean), firstk_mean, 'y');
%     elseif i==4
%         plot(1:length(firstk_mean), firstk_mean, 'b');
%     elseif i==5
%         plot(1:length(firstk_mean), firstk_mean, 'k');
%     elseif i==6
%         plot(1:length(firstk_mean), firstk_mean, 'm');
%     elseif i==7
%         plot(1:length(firstk_mean), firstk_mean, 'c');
%     else % i==8
%         plot(1:length(firstk_mean), firstk_mean, 'k--');
%     end
%     
% end



% shadedErrorBar(1:maxiter, mean(lower_bound_mat(:,:,1),2), std(lower_bound_mat(:,:,1),0, 2), 'g')
% shadedErrorBar(1:maxiter, mean(lower_bound_mat(:,:,2),2), std(lower_bound_mat(:,:,2),0, 2), 'r')
% shadedErrorBar(1:maxiter, mean(lower_bound_mat(:,:,3),2), std(lower_bound_mat(:,:,3),0, 2), 'b')
% shadedErrorBar(1:maxiter, mean(lower_bound_mat(:,:,4),2), std(lower_bound_mat(:,:,4),0, 2), 'g')
% shadedErrorBar(1:maxiter, mean(lower_bound_mat(:,:,5),2), std(lower_bound_mat(:,:,5),0, 2), 'm')
% shadedErrorBar(1:maxiter, mean(lower_bound_mat(:,:,6),2), std(lower_bound_mat(:,:,6),0, 2), 'r')
