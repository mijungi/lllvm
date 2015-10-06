% compute E

%% Ahmad's comment: this looks like it can definitely be optimized,
%%                  but is not used in training.

function [E, E_without_gamma] = compute_E(G, invV, mean_c, mean_x, nx, ny, n)

C = reshape(mean_c, ny, nx, n);
X = reshape(mean_x, nx, n);

Emat = zeros(n*ny,n);

for i=1:n
    
    Ci = C(:,:,i);
    Xi = X(:,i);
    
    
    for j=1:n
        
        eta_ij = G(i, j);
        
        if eta_ij==1
            
            Cj = C(:,:,j);
            Xj = X(:,j);
            
            
            Emat((i-1)*ny + 1: i*ny, j) = invV*(Cj+Ci)*(Xi-Xj);
            
        end
        
    end
    
end

E = sum(Emat,2);
E_without_gamma = 1/invV(1,1)*E;
