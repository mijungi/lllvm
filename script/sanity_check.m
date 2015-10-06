% to check new formulation with epsilon for adding epsilon*1*1'
% mijung wrote on Oct 1, 2015

clear all;
clc;
close all;

dx = 2; % dim(x)
dy = 3; % dim(y)
n = 20;  % number of datapoints

alpha = 1; % precision of X (zero-centering)
gamma = .2; % noise precision in likelihood
epsilon = 0.001; 

%% (1) define a graph

howmanyneighbors = 5;

% adjacency matrix
G = zeros(n,n,howmanyneighbors-1);

for i=1:howmanyneighbors-1
    G(:,:,i) = diag(ones(n-i,1),i);
    G(:,:,i) = G(:,:,i) + G(:,:,i)';
end

G = sum(G,3);

h = sum(G,2);
% laplacian matrix
L = diag(h) - G;

%% (2) generate x and C

% invOmega = kron(2*L,speye(dx));
% x = mnormrnd(zeros(dx, n), 1, inv(alpha*eye(n) + 2*L) );
% X = x(:);
% 
% U = eye(dy);
% ep_invOmega = epsilon*eye(size(invOmega)) + invOmega;
% v = mnormrnd(zeros(dy, dx*n), U, inv(ep_invOmega) );
% C = v(:);
% 
% %% (3) generate y
% 
% invV = gamma*speye(dy);
% V_y = inv(epsilon*eye(n) + 2*L*gamma);
% E = generateE(G, reshape(C,dy,n*dx), invV, X(:));
% E = reshape(E, dy, n);
% mu_y = E*V_y;
% Y = mnormrnd(mu_y, 1, V_y);
% y = Y(:);

%% check prior on x

x = randn(dx, n);

One_vec_n = ones(n,1);
invOmega = kron(2*L,speye(dx));

log_prob_x = log(mvnpdf(x(:)', zeros(1, n*dx), kron(inv(alpha*eye(n) + 2*L) , eye(dx))))

invPi = alpha*eye(n*dx) + invOmega;
exponent = -0.5*x(:)'*invPi*x(:); 

dist = zeros(n, n);
for i=1:n
    for j=1:n
        eta = G(i,j);
        if eta~=0
            dist(i,j) = 0.5*sum((x(:,i) - x(:,j)).^2);
        end
    end
end
exponent_eq = -0.5*alpha*sum(sum(x.^2)) - sum(sum(dist));

log_prob_x_eq = exponent_eq -0.5*n*dx*log(2*pi) + 0.5*logdetns(invPi);

[exponent exponent_eq abs(exponent-exponent_eq)]
[log_prob_x log_prob_x_eq abs(log_prob_x-log_prob_x_eq)]

%% check prior on C (we don't need beta, so remove it from the model)

v = randn(dy, dx*n);

J = kron(ones(n,1), eye(dx));

log_prob_c = log(mvnpdf(v(:)', zeros(1, n*dx*dy), kron(inv(epsilon*J*J' + invOmega), eye(dy))));
exponent = -epsilon*0.5*trace(v*J*J'*v') - 0.5*trace(invOmega*v'*v);

dist = zeros(n, n);
for i=1:n
    for j=1:n
        [i j]
        eta = G(i,j);
        if eta~=0
            dist(i,j) = 0.5*sum(sum((v(:,1+(i-1)*dx:i*dx) - v(:,1+(j-1)*dx:j*dx)).^2));
        end
    end
end

exponent_eq = - sum(sum(dist)) -0.5*epsilon* sum(sum((v*J).^2));
log_prob_c_eq = exponent  - 0.5*n*dx*dy*log(2*pi) + 0.5*dy*logdetns(epsilon*J*J' + invOmega);

[exponent exponent_eq]
[log_prob_c log_prob_c_eq]

%% check likelihood

y = randn(dy, n);

ones_n = ones(n,1);

% [-0.5*epsilon*sum(sum(y,2).^2)  -0.5*y(:)'*kron(epsilon*ones_n*ones_n', eye(dy))*y(:)]

invV = gamma*eye(dy);
Cmat = reshape(v, dy, dx, n);
E = compute_E(G, invV, Cmat, x, dx, dy, n);
E = reshape(E, dy, n);

M_y = E/(epsilon*ones_n*ones_n' + 2*gamma*L); 
exponent = - 0.5*trace(inv(epsilon*ones_n*ones_n' + 2*gamma*L)*(y-M_y)'*(y-M_y)); 

dist = zeros(n, n);
for i=1:n
    for j=1:n
        [i j]
        eta = G(i,j);
        if eta~=0
            lin_dist = y(:,j) - y(:,i) - v(:, 1+(i-1)*dx:i*dx)*(x(:,j) - x(:,i));
            dist(i,j) = 0.5*gamma*lin_dist'*lin_dist;
        end
    end
end

f = zeros(n,n);
for i=1:n
    for j=1:n
        eta = G(i,j);
        if eta~=0
            f(i,j) = gamma*((x(:,j) - x(:,i))'*v(:, 1+(i-1)*dx:i*dx)')*((x(:,j) - x(:,i))'*v(:, 1+(i-1)*dx:i*dx)')';
        end
    end
end
f = sum(sum(f));

sig_y = inv(kron(epsilon*ones_n*ones_n' + gamma*2*L, eye(dy)));
mu_y = sig_y*E(:); 

% [y(:)'*E(:) (mu_y'/sig_y)*y(:)]
% [f (mu_y'/sig_y)*mu_y]
exponent_eq_14 = - sum(sum(dist))-0.5*epsilon*sum(sum(y,2).^2);
exponent_eq_15 =  -0.5*((y(:)'/sig_y)*y(:)-2*y(:)'*E(:)+f);
[exponent_eq_14 exponent_eq_15]

log_prob_y = log(mvnpdf(y(:)', mu_y', sig_y));

logZy = 0.5*((mu_y'/sig_y)*mu_y - f) + 0.5*logdetns(2*pi*sig_y);
log_prob_y_eq = exponent_eq_15 -logZy;
[log_prob_y log_prob_y_eq]





