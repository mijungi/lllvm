% compute 

function QhatLtildeQhat = QhatLhatQhat(G, EXX, Ltilde) 


% computing Gamma using Qhat and Ltilde_epsilon (or Ltilde_L)
n = size(G,1);
dx = size(EXX,1)/n;
Gamma = zeros(n*dx, n*dx); 

% computing the upper off-diagonal first 
for i=1:n
    
%     j_nonzero_idx = find(G(i,:));
%     j_nonzero_idx =  j_nonzero_idx(logical(j_nonzero_idx>i));
%     for jj=1:length(j_nonzero_idx)
%         j = j_nonzero_idx(jj);
    
     for j=i+1:n
        
        nonzero_p = find(G(i,:));
        nonzero_q = find(G(j,:));
        
        sum_EXX = compute_sum_EXX(nonzero_p, nonzero_q, i, j, Ltilde, EXX); 

        Ltilde_pq = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, EXX, 'pq');
        Ltilde_pj = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, EXX, 'pj');
        Ltilde_iq = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, EXX, 'iq');
        Ltilde_ij = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, EXX, 'ij');
        
        EXX_cell_for_sum = mat2cell(Ltilde_pq.*sum_EXX, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
        EXX_mat_for_sum = reshape([EXX_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
        Ltilde_EXX_pq = sum(EXX_mat_for_sum, 3);
        
        EXX_cell_for_sum = mat2cell(Ltilde_pj.*sum_EXX, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
        EXX_mat_for_sum = reshape([EXX_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
        Ltilde_EXX_pj = sum(EXX_mat_for_sum, 3);
        
        EXX_cell_for_sum = mat2cell(Ltilde_iq.*sum_EXX, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
        EXX_mat_for_sum = reshape([EXX_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
        Ltilde_EXX_iq = sum(EXX_mat_for_sum, 3);
        
        EXX_cell_for_sum = mat2cell(Ltilde_ij.*sum_EXX, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
        EXX_mat_for_sum = reshape([EXX_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
        Ltilde_EXX_ij = sum(EXX_mat_for_sum, 3);

        Gamma(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = (Ltilde_EXX_pq - Ltilde_EXX_pj - Ltilde_EXX_iq + Ltilde_EXX_ij);
    end
end

Gamma = Gamma + Gamma'; 

% now computing the diagonal term

for i=1:n
    j=i;
    nonzero_p = find(G(i,:));
    nonzero_q = find(G(j,:));
    
    sum_EXX = compute_sum_EXX(nonzero_p, nonzero_q, i, j, Ltilde, EXX);
    
    Ltilde_pq = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, EXX, 'pq');
    Ltilde_pj = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, EXX, 'pj');
    Ltilde_iq = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, EXX, 'iq');
    Ltilde_ij = compute_Ltilde_CC(nonzero_p, nonzero_q, i, j, Ltilde, EXX, 'ij');
    
    EXX_cell_for_sum = mat2cell(Ltilde_pq.*sum_EXX, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
    EXX_mat_for_sum = reshape([EXX_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
    Ltilde_EXX_pq = sum(EXX_mat_for_sum, 3);
    
    EXX_cell_for_sum = mat2cell(Ltilde_pj.*sum_EXX, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
    EXX_mat_for_sum = reshape([EXX_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
    Ltilde_EXX_pj = sum(EXX_mat_for_sum, 3);
    
    EXX_cell_for_sum = mat2cell(Ltilde_iq.*sum_EXX, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
    EXX_mat_for_sum = reshape([EXX_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
    Ltilde_EXX_iq = sum(EXX_mat_for_sum, 3);
    
    EXX_cell_for_sum = mat2cell(Ltilde_ij.*sum_EXX, dx*ones(length(nonzero_p),1), dx*ones(length(nonzero_q),1));
    EXX_mat_for_sum = reshape([EXX_cell_for_sum{:}], dx, dx, length(nonzero_p)*length(nonzero_q));
    Ltilde_EXX_ij = sum(EXX_mat_for_sum, 3);
    
    Gamma(1+(i-1)*dx:i*dx, 1+(j-1)*dx:j*dx) = (Ltilde_EXX_pq - Ltilde_EXX_pj - Ltilde_EXX_iq + Ltilde_EXX_ij);
    
end

QhatLtildeQhat = Gamma; 