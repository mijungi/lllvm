
function test_whiskies_data(seed, k)

% set a seed 
oldRng = rng();
rng(seed);

% load data
load ../real_data/whisky.mat

% define essential quantities
Y =  whiskydata' ; 
% mean_shifted = bsxfun(@minus, whiskydata, mean(whiskydata,2));
% var_1 = bsxfun(@times, mean_shifted, 1./std(mean_shifted, 0,2));
% % Y = var_1;
% Y = mean_shifted + 0*randn(size(mean_shifted)); % adding noise
% Y = Y'; 
Y = normdata(Y);

n = size(Y, 2);
dy = size(Y,1);
dx = 2;

[Gtrue, dmat] = makeG(whiskycoords', n,2, k);
[sorted_val, sort_idx] = sort(dmat(72,:)); % the top north one is the center of distances
col = sorted_val;

figure(1)
subplot(231);
scatter( whiskycoords(sort_idx,1),  whiskycoords(sort_idx,2), 40, col, 'o', 'filled');
subplot(234);
% imagesc(Gtrue);title('G true');
gplot(Gtrue, whiskycoords, '*-');

% Y = Y(:,sort_idx); 
% G = makeG(Y, n,dy, k);

%%%%%%%%%%%%%%%%%%%
G = Gtrue; 
%%%%%%%%%%%%%%%%%%%

% subplot(235);
% % imagesc(Gtrue);title('G true');
% gplot(G, whiskycoords, '*-');


h = sum(G,2);
L = diag(h) - G; % laplacian matrix
% Add a scaled identity to L so that invOmega (part of precision of p(x) ) is
% not low rank.
L = L + 1e-3*eye(n);
invOmega = kron(2*L,eye(dx));

epsilon_1 = 0; % constant term added to prior precision for C
epsilon_2 = 0; % constant term added to likelihood for y


%% (2) EM

%% GPLVM


mappedX = gplvm(Y', dx);
subplot(233); scatter(mappedX(sort_idx,1), mappedX(sort_idx,2), 40, col, 'o', 'filled');
title('GPLVM');
Gest_gplvm = makeG(mappedX', n,2, k);
subplot(236);
% imagesc(Gtrue);title('G true');
gplot(Gest_gplvm, whiskycoords, '*-');

%%
diagU = true;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% (1) initialization of hyperparameters
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha_init = rand;
invPi_init = alpha_init*eye(size(invOmega)) + invOmega;
if diagU
    beta_init = rand;
    invU_init =  beta_init*eye(dy);
else
    invU_init = diag(rand(dy,1));
end

gamma_init = rand;
invV_init = gamma_init*eye(dy);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % (1) initialization for mean_x and cov_x to random values
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cov_x_qC = inv(speye(size(invPi_init)) + invPi_init);
% mean_x_qC = randn(n*dx,1);
mean_x_qC = reshape(whiskycoords', [], 1)/10000;


% Wittawat: create const struct containing all constants (do not change over EM
% iterations) needed.
const = struct();
% eigenvalues of L. nx1 vector.
const.dx = dx;
const.dy = dy;
const.eigv_L = eig(L);
const.n = n;

maxcount = 100;
icount = 1;

variational_lwb = zeros(maxcount, 1);

% to store results every 10th iteration
meanCmat = zeros(n*dx*dy, maxcount/10);
meanXmat = zeros(n*dx, maxcount/10);
covCmat = zeros(n*dx, maxcount/10);
covXmat = zeros(n*dx, maxcount/10);
alphamat = zeros(maxcount/10,1);
betamat = zeros(maxcount/10,1);
gammamat  = zeros(maxcount/10,1);

%%
while (icount<=maxcount)
    
    fprintf(['Performing: ' num2str(icount) ' of ' num2str(maxcount) ' iterations\n'])
    
    %% (1) E step
    
    fprintf('E-step')
    tic;
    % 1.1 : compute mean and cov of c given x (eq.19)
    [invGamma_qX, H_qX] = compute_invGamma_qX_and_H_qX(G, mean_x_qC, cov_x_qC, Y, n, dy, dx);
    if diagU
        beta_init = invU_init(1,1);
        gamma_init =  invV_init(1,1);
        cov_c_qX = inv(gamma_init*invGamma_qX + beta_init*(epsilon_1*eye(size(invOmega)) + invOmega));
        mean_c_qX = invV_init*H_qX*cov_c_qX';
    else
        cov_c_qX = inv(kron(invGamma_qX, invV_init) + kron(epsilon_1*eye(size(invOmega)) +invOmega, invU_init));
        mean_c_qX = cov_c_qX*reshape(invV_init*H_qX, [],1);
    end
    
    % 1.2 : compute mean and cov of x given c (eq.16)
    [invA_qC, B_qC, invA_qC_without_gamma] = compute_invA_qC_and_B_qC(G, mean_c_qX, cov_c_qX, invV_init, Y, n, dy, dx, diagU);
    cov_x_qC = inv(invA_qC + invPi_init);
    mean_x_qC = cov_x_qC*B_qC;
    
    tt = toc;
    display(sprintf(' took %.3g seconds', tt));
    
    %% (2) M step
    
    fprintf('M-step')
    tic;
    [lwb_C , ~, newU] = Mstep_updateU(invU_init, mean_c_qX, cov_c_qX, ...
        invOmega, n, dx, dy, diagU, epsilon_1);
    [newAlpha, lwb_x] = Mstep_updateAlpha(const, invOmega, mean_x_qC, cov_x_qC);
    %[newGamma, lwb_likelihood] = Mstep_updateGamma(const, mean_c_qX, cov_c_qX, ...
    %    invGamma_qX, H_qX, n, dy, L, invV_init, diagU, epsilon_2, Y);
%     [~, D_without_gamma] = computeD(G, Y, invV_init, L, epsilon_2);
    [newGamma, lwb_likelihood] = Mstep_updateGamma(const, mean_c_qX, cov_c_qX, invGamma_qX, H_qX, n, dy, L, invV_init, diagU, epsilon_2, Y);
    
    display([newAlpha, newGamma]);
    
    tt = toc;
    display(sprintf(' took %.3g seconds', tt));
    
    % update parameters
    invU_init = inv(newU);
    invV_init = newGamma*eye(dy);
    invPi_init =  newAlpha*eye(size(invOmega)) + invOmega;
    
    %% (3) compute the lower bound
    
    variational_lwb(icount) = lwb_likelihood + lwb_C + lwb_x; % eq.(21)+(22)+(23)

    subplot(235); plot(1:icount, variational_lwb(1:icount), 'o-');
    reshaped_mean_x = reshape(mean_x_qC, dx, []);
%     num_pt_visu = size(Y,2);
    subplot(232); scatter(reshaped_mean_x(1,sort_idx), reshaped_mean_x(2,sort_idx), 40, col, 'o', 'filled'); title('LLLVM');
%     scatter3(reshaped_mean_x(1,1:num_pt_visu), reshaped_mean_x(2,1:num_pt_visu), reshaped_mean_x(3,1:num_pt_visu), 40, col(1:num_pt_visu), 'o', 'filled');
%     grid off;
%     [Gest, ~] = makeG(reshaped_mean_x, n,2, k);
%     subplot(235);
%     gplot(Gest, whiskycoords, '*-'); title('G from lllvm');
    pause(0.5);
    
    % store results in every 10th iteration (otherwise the datasize to
    % store is too large).
    if rem(icount, 10)==0
        fprintf('Storing results \n')
        meanCmat(:,icount/10) = mean_c_qX(:);
        meanXmat(:,icount/10) = mean_x_qC(:);
        covCmat(:,icount/10) = diag(cov_c_qX); % store only diag of cov, due to too large size!
        covXmat(:,icount/10) = diag(cov_x_qC); % store only diag of cov, due to too large size!
        alphamat(icount/10) = newAlpha;
        betamat(icount/10) = invU_init(1,1);
        gammamat(icount/10) = newGamma;
    end
    
    icount = icount + 1;

end


[Gest, ~] = makeG(reshaped_mean_x, n,2, k);
subplot(235);
gplot(Gest, whiskycoords, '*-'); title('G from lllvm');

%%
funcs = funcs_global(); 
filename = ['whiskies_k_' num2str(k) '_s_' num2str(seed)];
filename_with_directory = funcs.scriptSavedFile([filename '.mat']);
save(filename_with_directory, 'variational_lwb', 'meanCmat', 'meanXmat', 'alphamat', 'betamat', 'gammamat', 'covCmat', 'covXmat', 'col');
% save(filename, 'variational_lwb', 'meanCmat', 'meanXmat', 'alphamat', 'betamat', 'gammamat', 'covCmat', 'covXmat', 'permuted_val');
% change seed back
rng(oldRng);