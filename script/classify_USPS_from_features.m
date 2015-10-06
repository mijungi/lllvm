% classify USPS data from features selected by LL-LVM, GP-LVM, LLE, and
% ISOMAP

clear all;
clc;

seed = 0;

rng(seed);

num_pt_each_digit = 120;
true_label = [0*ones(num_pt_each_digit,1); 1*ones(num_pt_each_digit, 1); 2*ones(num_pt_each_digit, 1); 3*ones(num_pt_each_digit, 1); 4*ones(num_pt_each_digit, 1)];

%% load data

load ../real_data/usps_resampled.mat

% combine test/train datasets, and select digits corresponds to 0-4 only
y_raw = [train_patterns test_patterns];
y_raw_label = [train_labels test_labels];

% indices for each digit
idx_0 = y_raw_label(1, :) ==1;
idx_1 = y_raw_label(2, :) ==1;
idx_2 = y_raw_label(3, :) ==1;
idx_3 = y_raw_label(4, :) ==1;
idx_4 = y_raw_label(5, :) ==1;

digit_0 = y_raw(:, idx_0);
digit_1 = y_raw(:, idx_1);
digit_2 = y_raw(:, idx_2);
digit_3 = y_raw(:, idx_3);
digit_4 = y_raw(:, idx_4);

digit_0 = digit_0(:,1:num_pt_each_digit);
digit_1 = digit_1(:,1:num_pt_each_digit);
digit_2 = digit_2(:,1:num_pt_each_digit);
digit_3 = digit_3(:,1:num_pt_each_digit);
digit_4 = digit_4(:,1:num_pt_each_digit);

% mix them and choose 3000 digits
digit_0_to_4 = [digit_0 digit_1 digit_2 digit_3 digit_4];

Y = digit_0_to_4;

%% running
% (1) LL-LVM

% n = 1000;
% seed = 5; k = floor(n/4);
% seed = 3;
n = size(Y,2);
dx = 2;
% kmat = floor(n./[8, 6, 4, 3, 2, 1]);
k = n/4; seed =5;
% k = n/2; seed = 6;
funcs = funcs_global();
% filename = ['USPS_k_' num2str(k) '_s_' num2str(seed)];
filename = ['USPS_k=' num2str(k) '_dx=' num2str(dx) '_n=' num2str(n) '_s=' num2str(seed)];
filename_with_directory = funcs.scriptSavedFile([filename '.mat']);
load(filename_with_directory );
which_to_show = 2;
xx_lllvm = reshape(meanXmat(:,which_to_show), 2, []);


% funcs = funcs_global();
% filename = ['USPS_k=' num2str(k) '_dx=' num2str(dx) '_n=' num2str(n) '_s=' num2str(seed)];
% filename_with_directory = funcs.scriptSavedFile([filename '.mat']);
% load(filename_with_directory);
% which_to_show = 4;
% xx_lllvm = reshape(rec_vars.mean_xs(:,which_to_show), 2, []);

subplot(221); scatter(xx_lllvm(1,:), xx_lllvm(2,:), 20, true_label, 'o', 'filled');
title('LLLVM');

%%
% (2) GP-LVM
mappedX_gplvm = gplvm(Y', dx);
subplot(222); scatter(mappedX_gplvm(:,1), mappedX_gplvm(:,2), 20, true_label, 'o', 'filled');
title('gplvm');

% (3) ISOMAP
[mappedX_isomap, mapping_isomap] = isomap(Y', dx, k);
subplot(223); scatter(mappedX_isomap(:,1), mappedX_isomap(:,2), 20, true_label, 'o', 'filled');
title('isomap');

% (4) LLE
[mappedX_lle, mapping_lle] = lle(Y', dx, k);
subplot(224); scatter(mappedX_lle(:,1), mappedX_lle(:,2), 20, true_label(mapping_lle.conn_comp), 'o', 'filled');
title('LLE');

%%  extract indices for training and test points

howmanyfolds =5;
train_idx_mat = zeros(length(Y), howmanyfolds);
test_idx_mat = zeros(length(Y), howmanyfolds);
for i=1:howmanyfolds
    [train_idx_mat(:,i), test_idx_mat(:,i)] = crossvalind('Resubstitution', length(Y), [0.1,0.9]);
end

%%

err_lllvm = zeros(howmanyfolds,1);
err_gplvm = zeros(howmanyfolds, 1);
err_isomap = zeros(howmanyfolds, 1);
err_lle = zeros(howmanyfolds, 1);
% err_y = zeros(howmanyfolds, 1);

for i=1:howmanyfolds
    
    train_idx = logical(train_idx_mat(:,i));
    test_idx = logical(test_idx_mat(:,i));
    
    % true label is different for each dataset
    classification_err = @(est) length(find(true_label(test_idx) ~= est))/length(true_label(test_idx));
    
    %% 1nn on y
    
%     mdl = fitcknn(Y(:,train_idx)',true_label(train_idx)+1);
%     label = predict(mdl,Y(:,test_idx)');
%     est_class = label-1;
%     err_y(i) = classification_err(est_class);
    
    %% LL-LVM
    
    mdl = fitcknn(xx_lllvm(:,train_idx)',true_label(train_idx)+1);
    label = predict(mdl,xx_lllvm(:,test_idx)');
    est_class = label-1;
    err_lllvm(i) = classification_err(est_class);
    
    %% GP-LVM
   
    mdl = fitcknn(mappedX_gplvm(train_idx, :),true_label(train_idx)+1);
    label = predict(mdl,mappedX_gplvm(test_idx,:));
    est_class = label-1;
    err_gplvm(i) = classification_err(est_class);
    
    %% (3) ISOMAP
    
    mdl = fitcknn(mappedX_isomap(train_idx, :),true_label(train_idx)+1);
    label = predict(mdl,mappedX_isomap(test_idx,:));
    est_class = label-1;
    err_isomap(i) = classification_err(est_class);
    
    %% (4) LLE
    
    mdl = fitcknn(mappedX_lle(train_idx, :),true_label(train_idx)+1);
    label = predict(mdl,mappedX_lle(test_idx,:));
    est_class = label-1;
    err_lle(i) = classification_err(est_class);
    
end



fprintf(['prediction error:  lllvm = ' num2str(mean(err_lllvm)) ' , gplvm =  ' num2str(mean(err_gplvm)), 'lle =  ' num2str(mean(err_lle)) ' , isomap =  ' num2str(mean(err_isomap))  '\n'])

%% visualisation of 10 fold cross validation results

% errorbar([mean(err_lllvm), mean(err_gplvm)],  [std(err_lllvm), std(err_gplvm)]);

errorbar([mean(err_lllvm), mean(err_isomap), mean(err_gplvm), mean(err_lle)],  [std(err_lllvm), std(err_isomap), std(err_gplvm),  std(err_lle)]);
% legend(['LL-LVM', 'GP-LVM', 'LLE'])

%%
barwitherr([std(err_lllvm), std(err_isomap), std(err_gplvm),  std(err_lle)], [mean(err_lllvm), mean(err_isomap), mean(err_gplvm), mean(err_lle)]);
% set(gca, 'x ['LLLVM','ISOMAP', 'GPLVM', 'LLE']);
% legend('LLLVM', 'ISOMAP', 'GPLVM', 'LLE')

