%% compute - Dkl(q(c)||p(c))
% mijung wrote on the 15th of Oct, 2015

function lwb_C = negDkl_C(mean_c, cov_c, invOmega, J, epsilon) 

% Inputs:
%  mean_c: dy x n*dx
%  cov_c: n*dx x n*dx
%  invOmega: n*dx x n*dx
% J : kron( ones_n , I_dx), hence, n*dx by n*dx 
% epsilon : a term added to eign val


% essential quantities
dy = size(mean_c,1);
ndx = size(cov_c,1);

epJJinvOmega = epsilon*J*J' + invOmega;
covCepJJinvOmega  = cov_c*epJJinvOmega;

logdetepJJ = logdetns(epJJinvOmega);

%  try
logdetcovc =  logdetns(cov_c) ;
%  catch
%      [u,d,v] = svd(cov_c);
%      diagd = diag(d);
%      threshold = 1e-6;
%      d_above_threshold = diagd(diagd>threshold);
%      logdetcovc =  sum(log(d_above_threshold)) ;
%  end
 
 logdettrm = logdetepJJ + logdetcovc;

lwb_C = 0.5*dy*logdettrm - 0.5*dy*trace(covCepJJinvOmega) + ...
    0.5*ndx*dy - 0.5*trace(epJJinvOmega*mean_c'*mean_c); 
