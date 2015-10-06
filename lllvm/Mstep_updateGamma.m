function [newGamma, objv] = Mstep_updateGamma(const, mean_c, cov_c, invGamma, H, n, ny, L, invV, diagU, epsilon_2, y, D_without_gamma)

% this function computes
% eq(21) = E_{q(c)q(x)} [ -0.5*Tr(invGamma*C'*invV*C - 2*C'*invV*H) - 0.5*D - log Zy ]
% where log Zy = 0.5*logdet(2*pi*Sigma_y)
%
% and find newGamma = argmax_{\gamma} eq(21)
%
% Input:
% mean_c: dy x n*dx
% cov_c: n*dx x n*dx
% invGamma: n*dx x n*dx (block diagonal). Each block is dx x dx. invGamma is NOT symmetric.
% H: dy x n*dx 
% invV: dy x dy. Inverse covariance in the likelihood. Scaled identity.
% diagU: true if U is a scaled identity beta*I.
% epsilon_2: an epsilon added to the Laplacian L so det(L) is not 0.
% y: dy x n
% L: n x n
%
% TODO: invGamma should be stored as a cell array of the diagonal blocks.
%

if diagU % if U is a diagonal matrix
    
    if nargin>12
    %exponent = E_{q(c)q(x)} [ -0.5*Tr(invGamma*C'*invV*C - 2*C'*invV*H) - 0.5*D
    %exponent_without_gamma = exponent/gamma;
   
    %exponent_without_gamma = trace(invGamma*(mean_c'*mean_c))+ ny*trace(invGamma*cov_c) - trace(2*mean_c'*H) + D_without_gamma;
    mean_c2 = mean_c'*mean_c;
    invGammaT = invGamma';
    exponent_without_gamma =  mean_c2(:)'*invGamma(:)+ ny*invGammaT(:)'*cov_c(:)...
        - 2*mean_c(:)'*H(:) + D_without_gamma;
    %exponent_without_gamma = mean_c2(:)'*invGamma(:) + ny*cov_c(:)'*invGamma(:)...
        %- 2*mean_c(:)'*H(:) + D_without_gamma;

    newgamma_inv = exponent_without_gamma/(ny*n);
    newGamma = 1/newgamma_inv;
    assert(newGamma > 0, 'updated gamma is not positive');

    normalizer_trm = 0.5*(- n*ny*log(2*pi) + ny*logdetns(2*L) + n*ny*log(newGamma));

    objv = -0.5*newGamma*exponent_without_gamma + normalizer_trm; 
    return;

    else
%     if false
        %frst_trm_D = epsilon_2*trace(y'*y);
        frst_trm_D = epsilon_2*y(:)'*y(:);
        tr_Lyty = compute_tr_Lyty(L, y);
        mean_c_invGamt = mean_c*invGamma';
        % compute trace(invGamma*(mean_c'*mean_c))
        tr_invGam_mct_mc = mean_c(:)'*mean_c_invGamt(:);
        % compute trace(invGamma*cov_c)
        %     tr_invGam_cov_c = cov_c(:)'*invGamma(:);
        tr_invGam_cov_c = reshape(invGamma', 1, [])*cov_c(:); 
        % compute trace(mean_c'*H)
        tr_mean_ct_H = mean_c(:)'*H(:);

        eigL = const.eigv_L; 

%         gammaTot = 2.^[-6:0.5:6];

%%%%%%%%%%%%%%%%%%%%%%
       % I am not optimising gamma
        gammaTot = const.gamma;
%%%%%%%%%%%%%%%%%%%%%%
       
        obj = zeros(length(gammaTot),1);

        for i=1:length(gammaTot)

            gamma = gammaTot(i);

            %         invV = gamma*speye(ny);
            %         D = frst_trm_D +  2 * sum(sum(L .* (y'*invV*y)));
            D = frst_trm_D +  2 *gamma*tr_Lyty ;
            exponent_trm = -0.5*(gamma*tr_invGam_mct_mc+ gamma*ny*tr_invGam_cov_c - 2*gamma*tr_mean_ct_H + D);

            logdettrm = sum(log(epsilon_2 + 2*gamma*eigL)); 
            normalizer_trm = 0.5*(- n*ny*log(2*pi) + ny*logdettrm);
            %          normalizer_trm = 0.5*(- n*ny*log(2*pi) + ny*logdetns(epsilon_2*eye(size(L)) + 2*gamma*L));


            obj(i) = exponent_trm + normalizer_trm ; % eq(21)
        end

        % plot(gammaTot, obj);

        [objv,v] = max(obj);
        newGamma = gammaTot(v);
    end
    
%     semilogx(gammaTot, obj, '--', gammaTot(v), objv, 'ro')
    
else
    
    %% (1) exponent_without_gamma
    
    % exponent = E_{q(c)q(x)} [ -0.5*Tr(invGamma*C'*invV*C - 2*C'*invV*H) - 0.5*D
    % exponent_without_gamma = exponent/gamma;
    
    I_ny = speye(ny);
    invGamma_kron_Iny = kron(invGamma, I_ny);
    exponent_without_gamma = -0.5*(mean_c'*invGamma_kron_Iny*mean_c + invGamma_kron_Iny(:)'*cov_c(:) - 2*mean_c'*H(:)) ...
        -0.5*D_without_gamma;
    
    %% compute eq(1) as a function of gamma
    
    gammaTot = 2.^[-10:0.1:10];
    obj = zeros(length(gammaTot),1);
    
    for i=1:length(gammaTot)
        
        gamma = gammaTot(i);
        exponent = gamma*exponent_without_gamma;
        
        invV = gamma*eye(size(invV));
        invSigma_y = kron(2*L, invV);
        normalizer_trm = 0.5*logdetns(invSigma_y) - n*ny/2*log(2*pi);
        
        obj(i) = exponent + normalizer_trm ; % eq(21)
        
    end
    
    % plot(gammaTot, obj);
    
    [objv,v] = max(obj);
    newGamma = gammaTot(v);
    
    semilogx(gammaTot, obj, '--', gammaTot(v), objv, 'ro')
    
end

%% one could use the closed form update (eq.24)

% newGammaInv = 1/(ny*n)*(-2*exponent_without_gamma -2*normalizer_trm_3_without_gamma -2*normalizer_trm_2_without_gamma);
%
%

end % end function 

function val =  compute_tr_Lyty(L, y)
    % compute tr(L*y'*y). Same as sum(sum(L .* (y'*y))).
    % y: dy x n
    % L: n x n
    [dy, n] = size(y);
    if n <= dy 
        yty = y'*y;
        val = L(:)'*yty(:);
    else 
        yL = y*L;
        val = y(:)'*yL(:);
    end
end

