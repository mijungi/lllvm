%% compute -Dkl(q(x)||p(x))
% mijung wrote on the 15th of Oct, 2015
% wittwat wrote alpha update sometime before NIPS deadline

function [lwb_x, alpha] = negDkl_x_update_alpha(mean_x, cov_x, invOmega, eigv_L)

ndx = size(mean_x,1);
dx = ndx/size(eigv_L,1);

invOmegaQuad = mean_x(:)'*invOmega*mean_x(:);
mean_x2 = mean_x(:)'*mean_x(:);
tr_cov_x = trace(cov_x);

% alpha update
% min_obj = @(a) -( dx*sum(log(a + 2*eigv_L)) - a*tr_cov_x - a*mean_x2 );
% opt = struct();
% opt.TolX = 1e-4;
% [alpha] = fminbnd(min_obj, 1e-5, 500, opt);

% alpha = eigv_L(end-1);
alpha = 1e-3; 

% compute the lower bound with the updated alpha 
% lwb_x  = 0.5*logdetns(cov_x) + 0.5*logdetns(alpha*eye(ndx) + invOmega)  ...
%     - 0.5*alpha*tr_cov_x - 0.5*trace(invOmega*cov_x) + 0.5*ndx ...
%     - 0.5*alpha*mean_x2 - 0.5*invOmegaQuad;

%  try
logdettrm =  logdetns(cov_x); 
%  catch
%      [u,d,v] = svd(cov_x);
%      diagd = diag(d);
%      threshold = 1e-6;
%      d_above_threshold = diagd(diagd>threshold);
%      logdettrm =  sum(log(d_above_threshold));
%  end

lwb_x  = 0.5*logdettrm + 0.5*dx*sum(log(alpha + 2*eigv_L))   ...
    - 0.5*alpha*tr_cov_x - 0.5*trace(invOmega*cov_x) + 0.5*ndx ...
    - 0.5*alpha*mean_x2 - 0.5*invOmegaQuad;
