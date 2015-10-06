close all;

%%% change : dx=2, or 4 %%%
dx = 2; 
%%%%%%%%%%%%%%%%%

n = 600;
% kmat = floor(n./[8, 6, 4, 3, 2]);
kmat = n./[8, 4, 2];
seedmat = 1:10;

maxiter = 20; 
lower_bound_mat = zeros(maxiter, length(seedmat), length(kmat));

for k_idx = 1:length(kmat)
    
    k = kmat(k_idx);
    
    for seed_idx = 1:length(seedmat)
        
        oldRng = rng();
        seed = seedmat(seed_idx);
        rng(seed);
        
        fprintf(['Testing: k= ' num2str(k) ' and seed = ' num2str(seed) '\n'])
        
        %% for color coding, check how I indexed datapoints:
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
        
        num_pt_each_digit =120;
        
        % take the first num_pt_each_digit datapoints
        non_zero_idx_0 = find(idx_0);
        digit_0_idx = non_zero_idx_0(1:num_pt_each_digit);
        non_zero_idx_1 = find(idx_1);
        digit_1_idx = non_zero_idx_1(1:num_pt_each_digit);
        non_zero_idx_2 = find(idx_2);
        digit_2_idx = non_zero_idx_2(1:num_pt_each_digit);
        non_zero_idx_3 = find(idx_3);
        digit_3_idx = non_zero_idx_3(1:num_pt_each_digit);
        non_zero_idx_4 = find(idx_4);
        digit_4_idx = non_zero_idx_4(1:num_pt_each_digit);
        
        % mix them and choose
        digit_val =  [0*ones(size(digit_0_idx)) ...
            1*ones(size(digit_1_idx))...
            2*ones(size(digit_2_idx))...
            3*ones(size(digit_3_idx))...
            4*ones(size(digit_4_idx))];
        digit_0_to_4_idx = [digit_0_idx digit_1_idx digit_2_idx digit_3_idx digit_4_idx];

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
        
        permuted_val = digit_val;
        Y = digit_0_to_4;
        
%         % (1) load the file:
%         filename = ['USPS_k_' num2str(k) '_s_' num2str(seed)];
%         load(filename);
        funcs = funcs_global(); 
%         filename = ['USPS_k=' num2str(k) '_n=' num2str(n) '_s=' num2str(seed)];
        filename = ['USPS_k=' num2str(k) '_dx=' num2str(dx) '_n=' num2str(n) '_s=' num2str(seed)];
        filename_with_directory = funcs.scriptSavedFile([filename '.mat']);
        load(filename_with_directory);
        lower_bound_mat(:,seed_idx,k_idx) = variational_lwb; 
%         lower_bound_mat(:,seed_idx,k_idx) = rec_vars.lwbs;
%         
%         % (2) visualize results:
%         figure(k)
%         hold on;
%         plot(variational_lwb);
        
        which_to_show = 2; 
        xx = reshape(meanXmat(:,which_to_show), 2, []);
%         which_to_show = 3; 
%         xx = reshape(rec_vars.mean_xs(:,which_to_show), 2, []);

%         figure(k+1)
%         subplot(221); scatter(xx(1,:), xx(2,:), 20, permuted_val, 'o', 'filled');
%         title('LL-LVM')
%         pause(0.5);
        %         figure(2); plot(meanXmat(
        
        %% wonder how LLE works with the same data
        
        %        digit_0 = y_raw(:, idx_0);
        %        digit_1 = y_raw(:, idx_1);
        %        digit_2 = y_raw(:, idx_2);
        %        digit_3 = y_raw(:, idx_3);
        %        digit_4 = y_raw(:, idx_4);
        %
        %        digit_0 = digit_0(:,1:600);
        %        digit_1 = digit_1(:,1:600);
        %        digit_2 = digit_2(:,1:600);
        %        digit_3 = digit_3(:,1:600);
        %        digit_4 = digit_4(:,1:600);
        %
        %        % mix them and choose 3000 digits
        %        digit_0_to_4 = [digit_0 digit_1 digit_2 digit_3 digit_4];
        %        Y = digit_0_to_4(:, ind_rand);
        %        dx = 2;
        %% LLE
        
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
        
%         pause;
        
        rng(oldRng);
        
    end
    
    
end

  
%% finding y* given x* using k neighbors

% % k = n/4; 
% % seedmat = 9;
% 
% % x_star = -0.2*rand(2,1);
% dx = 2;
% dy = size(Y,1);
% % x_star = [-0.5; -0.2] + randn(2,1)*0.05;
% x_str_idx = floor(n*rand);
% idx_chosen = ones(1,n);
% idx_chosen(x_str_idx) = 0; 
% 
% x_star = xx(:,x_str_idx);
% y_true = Y(:,x_str_idx);
% 
% % remove the last guy
% xx(:,x_str_idx) = [];
% C = reshape(meanCmat(:, which_to_show), dy, dx, n);
% C(:, :, x_str_idx) =[];
% Y(:, x_str_idx) =[];
% 
% diff_x = sum(bsxfun(@minus, xx, x_star).^2);
% [~, srt_id] = sort(diff_x, 'ascend');
% knei =k;
% id_selected = srt_id(1:knei);
% 
% Cselect = C(:,:,id_selected);
% Yselect = Y(:,id_selected);
% Xselect = xx(:,id_selected);
% 
% shift_x = bsxfun(@minus, x_star, Xselect);
% CtimesX = zeros(dy, knei);
% 
% for i=1:knei
%     CtimesX(:,i) = Cselect(:,:,i)*shift_x(:,i);
% end
% 
% y_star = reshape(mean(Yselect, 2), sqrt(dy), []) + reshape(mean(CtimesX,2), sqrt(dy), []);
% 
% figure(100)
% clf;
% subplot(221); scatter(xx(1,:), xx(2,:), 20, permuted_val(1:end-1), 'o', 'filled');hold on; 
% scatter(x_star(1), x_star(2), 40, 'ro', 'filled');
% title('LL-LVM')
% 
% subplot(222); imagesc(reshape(y_true, 16, [])'); title('true Y'); 
% % subplot(223); imagesc(reshape(mean(Yselect, 2), 16, [])'); title('mean of neighboring y');
% subplot(224); imagesc(y_star'); title('from our model'); 

%% lowerbound plotting

% k = n./[2, 4, 8];
% kmat = floor(n./[8, 6, 4, 3, 2, 1]);
hold on;
shadedErrorBar(1:maxiter, mean(lower_bound_mat(:,:,1),2), std(lower_bound_mat(:,:,1),0, 2), 'g')
shadedErrorBar(1:maxiter, mean(lower_bound_mat(:,:,2),2), std(lower_bound_mat(:,:,2),0, 2), 'r')
shadedErrorBar(1:maxiter, mean(lower_bound_mat(:,:,3),2), std(lower_bound_mat(:,:,3),0, 2), 'b')
% shadedErrorBar(1:maxiter, mean(lower_bound_mat(:,:,4),2), std(lower_bound_mat(:,:,4),0, 2), 'g')
% shadedErrorBar(1:maxiter, mean(lower_bound_mat(:,:,5),2), std(lower_bound_mat(:,:,5),0, 2), 'm')
% shadedErrorBar(1:maxiter, mean(lower_bound_mat(:,:,6),2), std(lower_bound_mat(:,:,6),0, 2), 'r')
