%% finding y* given x* using k neighbors for USPS data
% mijung wrote on Oct 26, 2015

clear all;
clc;
close all;

seed = 11;
rng(seed);

load Y_USPS_400.mat
Y = Y_USPS_400;

[dy, n] = size(Y);
dx = 2;
k  = floor(n/80);
seed = 5; 
funcs = funcs_global();
filename = ['fixing_alpha_USPS_k=' num2str(k) '_dx=' num2str(dx) '_n=' num2str(n) '_s=' num2str(seed)];

filename_with_directory = funcs.scriptSavedFile([filename '.mat']);
load(filename_with_directory );
which_to_show = length(results.lwbs);
xx_lllvm = reshape(results.mean_x(:,which_to_show), 2, []);


% xx = reshape(results.mean_x(:,which_to_show), 2, []);

for whichdigit = 0:4
    
    xx = xx_lllvm;
    
    x_str_idx = whichdigit*80 + randi(80);
    idx_chosen = ones(1,n);
    idx_chosen(x_str_idx) = 0;
    
    x_star = xx(:,x_str_idx);
    y_true = Y(:,x_str_idx);
    
    % remove the last guy
    xx(:,x_str_idx) = [];
    C = reshape(results.mean_c(:, which_to_show), dy, dx, n);
    C(:, :, x_str_idx) =[];
    Y(:, x_str_idx) =[];
    
    diff_x = sum(bsxfun(@minus, xx, x_star).^2);
    [~, srt_id] = sort(diff_x, 'ascend');
    knei =k;
    id_selected = srt_id(1:knei);
    
    Cselect = C(:,:,id_selected);
    Yselect = Y(:,id_selected);
    Xselect = xx(:,id_selected);
    
    shift_x = bsxfun(@minus, x_star, Xselect);
    CtimesX = zeros(dy, knei);
    
    for i=1:knei
        CtimesX(:,i) = Cselect(:,:,i)*shift_x(:,i);
    end
    
    y_star = reshape(mean(Yselect, 2), sqrt(dy), []) + reshape(mean(CtimesX,2), sqrt(dy), []);
    
    figure(whichdigit+1)
    subplot(221); scatter(xx(1,:), xx(2,:), 20, permuted_val(1:end-1), 'o', 'filled');hold on;
    scatter(x_star(1), x_star(2), 40, 'ro', 'filled');
    title('LL-LVM')
    
    subplot(222); imagesc(reshape(y_true, 16, [])'); title('true Y');
    % subplot(223); imagesc(reshape(mean(Yselect, 2), 16, [])'); title('mean of neighboring y');
    subplot(224); imagesc(y_star'); title('from our model');
end

