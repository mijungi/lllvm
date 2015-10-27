clear;
clc;


%%% change : dx=2, or 4 %%%
dx = 2; 
%%%%%%%%%%%%%%%%%

% n = 600;
% kmat = floor(n./[8, 6, 4, 3, 2]);

n  = 400; 
kmat = n./[32, 16, 8, 4, 2];
seedmat = 1:10;

maxiter = 30; 
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
        funcs = funcs_global(); 
        filename = ['USPS_k=' num2str(k) '_dx=' num2str(dx) '_n=' num2str(n) '_s=' num2str(seed)];

        filename_with_directory = funcs.scriptSavedFile([filename '.mat']);
        load(filename_with_directory);
        
        lower_bound_mat(1:length(results.lwbs),seed_idx,k_idx) = results.lwbs; 
        
%%         
%         % (2) visualize results:
        subplot(221); plot(results.lwbs);
        which_to_show = 5; 
        xx = reshape(results.mean_x(:,which_to_show), 2, []);

        subplot(222); scatter(xx(1,:), xx(2,:), 20, permuted_val, 'o', 'filled');
        title('LL-LVM')
        pause(0.5);
        
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


firstk = lower_bound_mat(:,:,1);
firstk(firstk==0) = nan;
firstk_mean = nanmean(firstk,2);
firstk_std = nanstd(firstk, 0, 2);
shadedErrorBar(1:maxiter, firstk_mean, firstk_std, 'g')

firstk = lower_bound_mat(:,:,2);
firstk(firstk==0) = nan;
firstk_mean = nanmean(firstk,2);
firstk_std = nanstd(firstk, 0, 2);
shadedErrorBar(1:maxiter, firstk_mean, firstk_std, 'r')


firstk = lower_bound_mat(:,:,3);
firstk(firstk==0) = nan;
firstk_mean = nanmean(firstk,2);
firstk_std = nanstd(firstk, 0, 2);
shadedErrorBar(1:maxiter, firstk_mean, firstk_std, 'b')

firstk = lower_bound_mat(:,:,4);
firstk(firstk==0) = nan;
firstk_mean = nanmean(firstk,2);
firstk_std = nanstd(firstk, 0, 2);
shadedErrorBar(1:maxiter, firstk_mean, firstk_std, 'k')


% shadedErrorBar(1:maxiter, mean(lower_bound_mat(:,:,1),2), std(lower_bound_mat(:,:,1),0, 2), 'g')
% shadedErrorBar(1:maxiter, mean(lower_bound_mat(:,:,2),2), std(lower_bound_mat(:,:,2),0, 2), 'r')
% shadedErrorBar(1:maxiter, mean(lower_bound_mat(:,:,3),2), std(lower_bound_mat(:,:,3),0, 2), 'b')
% shadedErrorBar(1:maxiter, mean(lower_bound_mat(:,:,4),2), std(lower_bound_mat(:,:,4),0, 2), 'g')
% shadedErrorBar(1:maxiter, mean(lower_bound_mat(:,:,5),2), std(lower_bound_mat(:,:,5),0, 2), 'm')
% shadedErrorBar(1:maxiter, mean(lower_bound_mat(:,:,6),2), std(lower_bound_mat(:,:,6),0, 2), 'r')
