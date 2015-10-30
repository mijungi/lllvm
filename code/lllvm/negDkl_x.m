%% compute -Dkl(q(x)||p(x))
% mijung wrote on the 15th of Oct, 2015

function lwb_x = negDkl_x(mean_x, cov_x, invOmega, alpha)

ndx = size(mean_x,1);

invOmegaQuad = mean_x(:)'*invOmega*mean_x(:);
mean_x2 = mean_x(:)'*mean_x(:);
tr_cov_x = trace(cov_x);

lwb_x  = 0.5*logdetns(cov_x) + 0.5*logdetns(alpha*eye(ndx) + invOmega)  ...
    - 0.5*alpha*tr_cov_x - 0.5*trace(invOmega*cov_x) + 0.5*ndx ...
    - 0.5*alpha*mean_x2 - 0.5*invOmegaQuad;