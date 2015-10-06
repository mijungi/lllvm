function [D, D_without_gamma] = computeD(G, y, invV, L, epsilon_2)
%function [D, D_without_gamma] = computeD(G, y, invV, n)

% D = zeros(n,n);
% D_without_gamma = zeros(n,n);

% for i=1:n
%     for j=1:n
        
%         eta_ij = G(i,j);
        
%         if eta_ij==1
            
%             yi = y(:,i);
%             yj = y(:,j);
            
%             D(i,j) = eta_ij*(yj-yi)'*invV*(yj-yi);
%             D_without_gamma(i,j) = eta_ij*(yj-yi)'*(yj-yi);

%         end
        
%     end
% end

% D = sum(sum(D));
% D_without_gamma = sum(sum(D_without_gamma));

frst_trm_D = epsilon_2*trace(y'*y);
D = frst_trm_D +  2 * sum(sum(L .* (y'*invV*y)));
D_without_gamma = 2 * sum(sum(L .* (y'*y)));
