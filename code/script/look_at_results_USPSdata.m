% to visualise variational lower bound
% tested on USPS data

clear;
clc;

dx = 2; 

n  = 400;
kmat = floor(n./[100, 80, 50, 40]);
seedmat = 1:9;

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
%         filename = ['USPS_k=' num2str(k) '_dx=' num2str(dx) '_n=' num2str(n) '_s=' num2str(seed)];
        filename = ['fixing_alpha_USPS_k=' num2str(k) '_dx=' num2str(dx) '_n=' num2str(n) '_s=' num2str(seed)];

        filename_with_directory = funcs.scriptSavedFile([filename '.mat']);
        load(filename_with_directory);
        
        lower_bound_mat(1:length(results.lwbs),seed_idx,k_idx) = results.lwbs; 
        
%%         
%         % (2) visualize results:
%         subplot(221); plot(results.lwbs);
        which_to_show = length(results.lwbs); 
        xx = reshape(results.mean_x(:,which_to_show), 2, []);

        subplot(221); scatter(xx(1,:), xx(2,:), 20, permuted_val, 'o', 'filled');
        title('LL-LVM')
        
        rng(oldRng);
        
    end
    
    
end

%% lowerbound plotting

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

