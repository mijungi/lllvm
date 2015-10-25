function [results, op ] = lllvm_1ep(Y, op)
%LLLVM Perform dimensionality reduction with Locally Linear Latent Variable Model.
%
%   - Assume V^-1 = gamma*I (predicision matrix in the likelihood).
%Input:
% - Y: dy x n matrix of observed data.
% - op: a struct containing various options. See the code with lines containing
%   myProcessOptions or isfield for possible options.
%

% random seed. Default to 1.
seed = myProcessOptions(op, 'seed', 1);
oldRng = rng();
rng(seed);

[dy, n] = size(Y);
% center the data
Y = bsxfun(@minus, Y, mean(Y, 2));

% maximum number of EM iterations to perform.
max_em_iter = myProcessOptions(op, 'max_em_iter', 200);

% relative tolerance of the increase of the likelihood.
% If (like_i - like_{i-1} ) / like_{i-1}  < rel_tol, stop EM.
%rel_tol = myProcessOptions(op, 'rel_tol', 1e-3);

% absolute tolerance of the increase of the likelihood.
% If (like_i - like_{i-1} )  < abs_tol, stop EM.
abs_tol = myProcessOptions(op, 'abs_tol', 1e-4);

if ~isfield(op, 'G')
    error('option G for the neighbourhood graph is mandatory');
end
G = op.G;
assert(all(size(G) == [n,n]), 'Size of G must be n x n.');

% dimensionality of the reduced representation X. This must be less than dy.
% This option is mandatory.
if ~isfield(op, 'dx')
    error('option dx (reduced dimensionality) is mandatory.');
end
dx = op.dx;
assert(dx > 0);
%assert(dx <= dy, 'reduced dimension dx is larger than observations');

L = diag(sum(G, 1)) - G;
invOmega = kron(2*L, eye(dx));

% Intial value of alpha. Alpha appears in precision in the prior for x (low
% dimensional latent). This will be optimized in M steps.
alpha0 = myProcessOptions(op, 'alpha0', 1);
assert(alpha0 > 0, 'require alpha0 > 0');


% invPi is Pi^-1 where p(x|G, alpha) = N(x | 0, Pi) (prior of X).
invPi = alpha0*eye(n*dx) + invOmega;

% Initial value of gamma. V^-1 = gamma*I_dy where V is the covariance in the
% likelihood of  the observations Y.
gamma0 = myProcessOptions(op, 'gamma0', 1);
assert(gamma0 > 0, 'require gamma0 > 0');

% epsilon is a positive real number in the likelihood term specifying the amount 
% to penalize deviation of the mean of the data from 0. 
epsilon = myProcessOptions(op, 'epsilon', 1e-4);


% A recorder is a function handle taking in a struct containing all
% intermediate variables in LL-LVM in each EM iteration and does something.
% lllvm will call the recorder at the end of every EM iteration with a struct
% containing all variables if one is supplied.
%
% For example, a recorder may just print all the variables in every iteration.
recorder = myProcessOptions(op, 'recorder', []);

J = kron(ones(n,1), eye(dx));
% initial value of cov_c 
cov_c0 = myProcessOptions(op, 'cov_c0', inv(epsilon*(J*J') + invOmega) );

% initial value of mean_c
mean_c0 = myProcessOptions(op, 'mean_c0', randn(dy, dx*n)*cov_c0');


% collect all options used.
op.seed = seed;
op.max_em_iter = max_em_iter;
op.alpha0 = alpha0;
op.gamma0 = gamma0;
op.epsilon = epsilon;
% We will not collect the recorder as it will make a saved file big.


%========= Start EM
gamma_new = gamma0;
invPi_new = invPi;


opt_dec = 1; % using decomposition
[Ltilde] = compute_Ltilde(L, epsilon, gamma_new, opt_dec);
eigv_L = Ltilde.eigL;

cov_c = cov_c0;
mean_c = mean_c0;

% to store results
meanCmat = zeros(n*dx*dy, max_em_iter);
meanXmat = zeros(n*dx, max_em_iter);
covCmat = zeros(n*dx, max_em_iter);
covXmat = zeros(n*dx, max_em_iter);

alphamat = zeros(max_em_iter,1);
gammamat  = zeros(max_em_iter,1);

% vars to store lower bound
prev_lwb = inf();
% lower bound in every iteration
lwbs = [];
for i_em = 1:max_em_iter
    
    tic;
    %% (1) E step
        
    % compute mean and cov of x given c (eq.47 and 48)
    [A, b] = compute_suffstat_A_b(G, mean_c, cov_c, Y, gamma_new, epsilon);
    cov_x = inv(A+ invPi_new);
    mean_x = cov_x*b;
    
    % compute mean and cov of c given x (eq.56)
    [Gamma, H, Gamma_L]  = compute_suffstat_Gamma_h(G, mean_x, cov_x, Y, gamma_new, Ltilde);
    cov_c = inv(Gamma + epsilon*J*J' + invOmega);
    mean_c = gamma_new*H*cov_c';

    %% (2) M step
    [lwb_likelihood, gamma_new] = exp_log_likeli_update_gamma(mean_c, cov_c, H, Y, L, epsilon, Ltilde, Gamma_L);
    lwb_C = negDkl_C(mean_c, cov_c, invOmega, J, epsilon);
    [lwb_x, alpha_new] = negDkl_x_update_alpha(mean_x, cov_x, invOmega, eigv_L);
    
    % (2.half) update invPi using the new alpha
    invPi_new = alpha_new*eye(n*dx) + invOmega;
    
    %% (3) compute the lower bound
    
    lwb = lwb_likelihood + lwb_C + lwb_x;
    
    iter_time = toc();
    
    display(sprintf('EM iteration %d/%d. Took: %.3f s. Lower bound: %.3f.', ...
        i_em, max_em_iter, iter_time, lwb ));
    
    lwbs(end+1) = lwb;
    
    % call the recorder
    if ~isempty(recorder)
        % collect all the variables into a struct to pass to the recorder
        % take everything from the op except the recorder itself
        op2 = rmfield(op, 'recorder');
        state = op2;
        state.Y = Y;
        state.i_em = i_em;
        state.alpha = alpha_new;
        state.gamma = gamma_new;
        state.Gamma = Gamma;
        state.H = H;
        state.cov_c = cov_c;
        state.mean_c = mean_c;
        state.cov_x = cov_x;
        state.mean_x = mean_x;
        state.A = A;
        state.b = b;
        state.lwb = lwb;
        
        % call the recorder
        recorder(state);
    end
    
    % store results (all updated variables)
    meanCmat(:,i_em) = mean_c(:);
    meanXmat(:,i_em) = mean_x(:);
    covCmat(:,i_em) = diag(cov_c); % store only diag of cov, due to too large size!
    covXmat(:,i_em) = diag(cov_x); % store only diag of cov, due to too large size!

    alphamat(i_em) = alpha_new;
    gammamat(i_em) = gamma_new;

    
    % check increment of the lower bound.
    if i_em >= 2 && abs(lwb - prev_lwb) < abs_tol
        % stop if the increment is too low.
        break;
    end
    prev_lwb = lwb;
    
end %end main EM loop


% construct the results struct
r.cov_c = covCmat;
r.mean_c = meanCmat;
r.cov_x = covXmat;
r.mean_x = meanXmat;
r.alpha = alphamat;
r.gamma = gammamat;
r.lwbs = lwbs;
results = r;

rng(oldRng);
end

