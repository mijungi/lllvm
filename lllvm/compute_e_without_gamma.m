function e_k_l_trm_without_gamma = compute_e_without_gamma(k, l, ny, Sigma_y_without_gamma, mean_c, cov_c, mean_x, cov_x, G)

n = size(G,1);
e_k_l_trm_without_gamma = zeros(n, n);

nx = size(mean_x,1)/n;

Sigma_y_k_l_without_gamma = Sigma_y_without_gamma(1+(k-1)*ny:k*ny, 1+(l-1)*ny:l*ny); 

Cmat = reshape(mean_c, ny, nx, n);
Ck = Cmat(:,:,k);
Cl = Cmat(:,:,l);

xmat = reshape(mean_x, nx, n);
xk = xmat(:,k);
xl = xmat(:,l);

Sigxkl = cov_x(1+(k-1)*nx:k*nx, 1+(l-1)*nx:l*nx);

%%%%%%%%%%%%%%%%%%%%%%%
nxny = nx*ny;
Sigckl = cov_c(1+(k-1)*nxny:k*nxny, 1+(l-1)*nxny:l*nxny);
%%%%%%%%%%%%%%%%%%%%%%%

%%

for j=1:n
    
    Cj = Cmat(:,:,j);
    xj = xmat(:,j);
    
    Sigxjl = cov_x(1+(j-1)*nx:j*nx, 1+(l-1)*nx:l*nx);
    Sigcjl = cov_c(1+(j-1)*nxny:j*nxny, 1+(l-1)*nxny:l*nxny);
    
    eta_jk = G(j,k);
    
    %%
    if eta_jk ==1
        
        
        for u=1:n
%             [u n]
%          for u=j:n

            
            %         eta_jk = G(j,k);
            eta_ul = G(u,l);
            
            %         [j u]
            
            %         if (eta_jk==1)&&(eta_ul==1)
            
            if eta_ul == 1
                
                Cu = Cmat(:,:,u);
                xu = xmat(:,u);
                
                Sigxku = cov_x(1+(k-1)*nx:k*nx, 1+(u-1)*nx:u*nx);
                Sigxju = cov_x(1+(j-1)*nx:j*nx, 1+(u-1)*nx:u*nx);
                
                Sigcku = cov_c(1+(k-1)*nxny:k*nxny, 1+(u-1)*nxny:u*nxny);
                Sigcju = cov_c(1+(j-1)*nxny:j*nxny, 1+(u-1)*nxny:u*nxny);
                
                %% quqdratic terms
                
                % Trm1
                Sju = eta_jk*eta_ul*Cj'*Sigma_y_k_l_without_gamma*Cu;
                Trm1 = compute_quadratic_trm_in_e_k_l_trm_without_gamma(Sju, xk, xl, xj, xu, Sigxkl, Sigxku, Sigxjl, Sigxju);
                
                % Trm2
                Sjl = eta_jk*eta_ul*Cj'*Sigma_y_k_l_without_gamma*Cl;
                Trm2 = compute_quadratic_trm_in_e_k_l_trm_without_gamma(Sjl, xk, xl, xj, xu, Sigxkl, Sigxku, Sigxjl, Sigxju);
                
                % Trm3
                Sku = eta_jk*eta_ul*Ck'*Sigma_y_k_l_without_gamma*Cu;
                Trm3 = compute_quadratic_trm_in_e_k_l_trm_without_gamma(Sku, xk, xl, xj, xu, Sigxkl, Sigxku, Sigxjl, Sigxju);
                
                % Trm4
                Skl = eta_jk*eta_ul*Ck'*Sigma_y_k_l_without_gamma*Cl;
                Trm4 = compute_quadratic_trm_in_e_k_l_trm_without_gamma(Skl, xk, xl, xj, xu, Sigxkl, Sigxku, Sigxjl, Sigxju);
                
                %% trace terms
                
                Outerproduct_xk_minus_kj_xl_minus_ku = Sigxkl + xk*xl' - (Sigxku + xk*xu') - (Sigxjl + xj*xl') + (Sigxju + xj*xu');
                eta_ul_eta_jk_Sigma_y_k_l_without_gamma = eta_ul*eta_jk*Sigma_y_k_l_without_gamma;
                
                kronTrm = kron(Outerproduct_xk_minus_kj_xl_minus_ku, eta_ul_eta_jk_Sigma_y_k_l_without_gamma);
                
                sumTrm = Sigcju + Sigcjl + Sigcku + Sigckl;
                
                trTrm = trace(kronTrm*sumTrm);
                %             trTrm = kronTrm(:)'*sumTrm(:);
                
                %% sum all up
                
                e_k_l_trm_without_gamma(j,u) = Trm1 + Trm2 + Trm3 + Trm4 + trTrm;
                
                %         e_k_l_trm_without_gamma(j,u) = Trm1 + Trm2 + Trm3 + Trm4 + trTrm1 + trTrm2 + trTrm3 + trTrm4;
            end
            
        end
        
        %%
    end
    
end

%%
e_k_l_trm_without_gamma = sum(sum(e_k_l_trm_without_gamma));

%%
% lmat = triu(e_k_l_trm_without_gamma, 1)';
% umat = triu(e_k_l_trm_without_gamma);
% newresults = lmat + umat;
% 
% e_k_l_trm_without_gamma = sum(sum(newresults));