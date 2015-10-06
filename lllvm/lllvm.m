function [results, op ] = lllvm(Y, op)
%LLLVM Perform dimensionality reduction with Locally Linear Latent Variable Model.
%
%   - Assume V^-1 = gamma*I (predicision matrix in the likelihood).
%Input:
% - Y: dy x n matrix of observed data.
% - op: a struct containing various options. See the code with lines containing 
%   myProcessOptions or isfield for possible options.
%
%created: 21 May 2015
%

% random seed. Default to 1.
seed = myProcessOptions(op, 'seed', 1);
oldRng = rng();
rng(seed);

[dy, n] = size(Y);

% maximum number of EM iterations to perform.
max_em_iter = myProcessOptions(op, 'max_em_iter', 500);

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

% The factor to be multipled to an identity matrix to be added to the graph 
% Laplacian. This must be positive and typically small.
ep_laplacian = myProcessOptions(op, 'ep_laplacian', 1e-3);
assert(ep_laplacian > 0, 'ep_laplacian must be > 0');

L = diag(sum(G, 1)) - G + ep_laplacian*eye(n);
invOmega = kron(2*L, eye(dx));

% Intial value of alpha. Alpha appears in precision in the prior for x (low
% dimensional latent). This will be optimized in M steps.
alpha0 = myProcessOptions(op, 'alpha0', 1);
assert(alpha0 > 0, 'require alpha0 > 0');

% invPi is Pi^-1 where p(x|G, alpha) = N(x | 0, Pi) (prior of X).
invPi = alpha0*eye(n*dx) + invOmega;

% Initial value of beta. U^-1 = beta*I_dy where P(C) = MN(0, U, Omega).
beta0 = myProcessOptions(op, 'beta0', 1);
assert(beta0 > 0, 'require beta0 > 0');

% Initial value of gamma. V^-1 = gamma*I_dy where V is the covariance in the
% likelihood of  the observations Y.
gamma0 = myProcessOptions(op, 'gamma0', 1);
assert(gamma0 > 0, 'require gamma0 > 0');

% initial value of the posterior covariance cov_x of X. Size: n*dx x n*dx.
% Appear in q(x) = N(x | mean_x, cov_x) where x is the vectorized n*dx x 1 vector.
if isfield(op, 'cov_x0')
    cov_x0 = op.cov_x0;
else
    % compute the default cov_x0
    cov_x0 = inv(eye(n*dx) + invPi) + eye(n*dx);
end

% initial value of the posterior mean of X. Size: n*dx x 1.
% Appear in q(x) = N(x | mean_x, cov_x) where x is the vectorized n*dx x 1 vector.
mean_x0 = myProcessOptions(op, 'mean_x0', cov_x0*randn(n*dx, 1) );

% A recorder is a function handle taking in a struct containing all
% intermediate variables in LL-LVM in each EM iteration and does something. 
% lllvm will call the recorder at the end of every EM iteration with a struct
% containing all variables if one is supplied. 
%
% For example, a recorder may just print all the variables in every iteration.
recorder = myProcessOptions(op, 'recorder', []);

% collect all options used.
op.seed = seed; 
op.max_em_iter = max_em_iter;
op.abs_tol = abs_tol;
op.G = G;
op.dx = dx;
op.ep_laplacian = ep_laplacian;
op.alpha0 = alpha0;
op.beta0 = beta0;
op.gamma0 = gamma0;
op.cov_x0 = cov_x0;
op.mean_x0 = mean_x0;
% We will not collect the recorder as it will make a saved file big.

%  create const struct containing all constants that do not change over EM
%  iterations. 
const = struct();
% eigenvalues of L. nx1 vector.
const.dx = dx;
const.dy = dy;
const.eigv_L = eig(L);
const.n = n;

%========= Start EM 
alpha = alpha0;
beta = beta0;
invU = beta*eye(dy);
gamma = gamma0;
invV = gamma*eye(dy);

cov_x = cov_x0;
mean_x = mean_x0;

% This is a flag for legacy code. Not important.
diagU = true;
epsilon_1 = 0;
epsilon_2 = 0;

% vars to store lower bound 
prev_lwb = inf();
% lower bound in every iteration
lwbs = [];
for i_em = 1:max_em_iter
    
    %% (1) E step
    
    %fprintf('E-step')
    % measure time for one EM iteration.
    tic;
    % 1.1 : compute mean and cov of c given x (eq.19)
    [invGamma_qX, H_qX] = compute_invGamma_qX_and_H_qX(G, mean_x, cov_x, Y, n, dy, dx);
    beta = invU(1,1);
    gamma =  invV(1,1);
    cov_c = inv(gamma*invGamma_qX + beta*invOmega);
    mean_c = invV*H_qX*cov_c';
    
    % 1.2 : compute mean and cov of x given c (eq.16)
    [invA_qC, B_qC ] = compute_invA_qC_and_B_qC(G, mean_c, cov_c, invV, Y, n, dy, dx, diagU);
    cov_x = inv(invA_qC + invPi);
    mean_x = cov_x*B_qC;
    
    %% (2) M step
    
    %fprintf('M-step')
    [lwb_C , ~, U] = Mstep_updateU(invU, mean_c, cov_c, ...
        invOmega, n, dx, dy, diagU, epsilon_1);
    [alpha, lwb_x] = Mstep_updateAlpha(const, invOmega, mean_x, cov_x);
    %[gamma, lwb_likelihood] = Mstep_updateGamma(const, mean_c, cov_c, ...
    %    invGamma_qX, H_qX, n, dy, L, invV, diagU, epsilon_2, Y);
    [~, D_without_gamma] = computeD(G, Y, invV, L, epsilon_2);
    [gamma, lwb_likelihood] = Mstep_updateGamma(const, mean_c, cov_c, ...
        invGamma_qX, H_qX, n, dy, L, invV, diagU, epsilon_2, Y,  D_without_gamma);
    
    % update parameters
    % U is a scaled identity. U^-1 = beta*I_dy
    invU = eye(dy)/U(1,1);
    invV = gamma*eye(dy);
    invPi =  alpha*eye(size(invOmega)) + invOmega;
    
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
        state.alpha = alpha;
        state.beta = beta;
        state.gamma = gamma;
        state.invGamma_qX = invGamma_qX;
        state.H_qX = H_qX;
        state.cov_c = cov_c;
        state.mean_c = mean_c;
        state.cov_x = cov_x;
        state.mean_x = mean_x;
        state.invA_qC = invA_qC;
        state.B_qC = B_qC;
        state.lwb = lwb;

        % call the recorder 
        recorder(state);
    end
    
    % check increment of the lower bound.
    if i_em >= 2 && abs(lwb - prev_lwb) < abs_tol
        % stop if the increment is too low.
        break;
    end
    prev_lwb = lwb;

end %end main EM loop


% construct the results struct 
r = const;
r.cov_c = cov_c;
r.mean_c = mean_c;
r.cov_x = cov_x;
r.mean_x = mean_x;
r.alpha = alpha;
r.beta = beta;
r.gamma = gamma;
r.lwbs = lwbs;
results = r;

rng(oldRng);
end

