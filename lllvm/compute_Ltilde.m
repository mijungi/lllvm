% compute Ltilde by a chosen option
% (1) its definition
% (2) singular value decomposition of L
% mijung wrote on the 19th of Oct, 2015

function [result] = compute_Ltilde(L, epsilon, gamma, opt_dec)

% decompose Ltilde using singular value decomposition of L
% definition of  Ltilde = inv(epsilon*ones(n,1)*ones(1,n) + 2*gamma*L);
%  we can re-write Ltilde = Ltilde_epsilon + 1/(2*gamma)* Ltilde_L

n = size(L,1);

if opt_dec==0
    
    result = inv(epsilon*ones(n,1)*ones(1,n) + 2*gamma*L);
    
else
    
    [U_L, D_L, V_L] = svd(L); % L := U_L*D_L*V_L'
    
    Depsilon_inv = zeros(n,n);    
    sign_sin_val = V_L(:,end)'*U_L(:,end); % because matlab sometimes flips the sign of singular vectors
    Depsilon_inv(n,n) = sign_sin_val *1./(epsilon*n); 
    Ltilde_epsilon = V_L*Depsilon_inv*U_L' ;
    
    D_L_inv = zeros(n,n);
    D_L_inv(1:n-1, 1:n-1) = diag(1./diag(D_L(1:n-1, 1:n-1)));
    Ltilde_L = V_L*D_L_inv*U_L';
    
    % check if they match
    Ltilde = Ltilde_epsilon + 1./(2*gamma)* Ltilde_L;
    
    % return everything
    result.Ltilde = Ltilde; 
    result.Ltilde_epsilon = Ltilde_epsilon;
    result.Ltilde_L = Ltilde_L;
    result.eigL = diag(D_L); 
    
end

