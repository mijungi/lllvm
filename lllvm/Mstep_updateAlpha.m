function [newAlpha, obj_alpha] = Mstep_updateAlpha(const, invOmega, mean_x, cov_x)

    [newAlpha, obj_alpha] = Mstep_updateAlpha_mijung(const, invOmega, mean_x, cov_x);
%     [newAlpha, obj_alpha] = Mstep_updateAlpha_wittawat(const, invOmega, mean_x, cov_x);
end 

function [newAlpha, obj_alpha] = Mstep_updateAlpha_wittawat(const, invOmega, mean_x, cov_x)

%%
% tic;

ndx = size(mean_x,1);
eigv_L = const.eigv_L;
dx = const.dx;

%alpha = 2.^[-6:2:10];
%obj = zeros(length(alpha),1);
invOmegaQuad = mean_x(:)'*invOmega*mean_x(:);
%for i = 1:length(alpha)
    % The following lines are to check with Mijung's version.
    %invPi = alpha(i)*eye(ndx) + invOmega;
    %invOmega_times_cov_x = invOmega*cov_x;
    %invPicovx = alpha(i)*cov_x + invOmega_times_cov_x;
    %obj(i) = logdetns( invPicovx) - trace( invPicovx) +ndx - mean_x'*invPi*mean_x;

    %ob = logdetns(cov_x) + dx*sum(log(alpha(i) + 2*eigv_L)) - alpha(i)*trace(cov_x) ...
    %  -invOmega(:)'*cov_x(:) - alpha(i)*mean_x2 - invOmegaQuad + ndx;
    %display(sprintf('alpha obj. old ver: %.3g. new ver: %.3g', obj(i), ob));
%end
%[obj_alpha,v] = max(0.5*obj);
%newAlpha = alpha(v); % choose the maximum

mean_x2 = mean_x(:)'*mean_x(:);
tr_cov_x = trace(cov_x);
% find the root of the stationarity condition of alpha
%fun = @(a)0.5*(dx*sum(1.0./(a+eigv_L)) - tr_cov_x - mean_x2 );
%alpha0 = 1;
%% fzero may return a negative alpha which will yield an imaginary lower bound.
%newAlpha = fzero(fun, alpha0);
min_obj = @(a) -( dx*sum(log(a + 2*eigv_L)) - a*tr_cov_x - a*mean_x2 );
opt = struct();
opt.TolX = 1e-4;
% Wittawat: This line does not slow down the whole EM. 
% We do not need to sacrifice the tolerance or bounding box for speed.
% We should make them so that they are applicable to many things.
[newAlpha, negObj] = fminbnd(min_obj, 1e-5, 500, opt);

obj_alpha  = 0.5*(logdetns(cov_x) + dx*sum(log(newAlpha + 2*eigv_L)) - newAlpha*tr_cov_x...
      -invOmega(:)'*cov_x(:) - newAlpha*mean_x2 - invOmegaQuad + ndx );

% tt = toc;
% display(sprintf('Mstep_updateAlpha took: %.3g', tt));
% display(sprintf('Optimized alpha: %.3g. Func val: %.5g', newAlpha, obj_alpha));

end


function [newAlpha, obj_alpha] = Mstep_updateAlpha_mijung(const, invOmega, mean_x, cov_x)

I = speye(size(invOmega,1));

%%
% tic;
% alpha = 2.^[-6:2:10];

%%%%%%%%%%%%%%%%%%%%%%
% I am not optimising alpha
alpha = const.alpha;
%%%%%%%%%%%%%%%%%%%%%%

%alpha = 2.^[-2:0.005:2];

obj = zeros(length(alpha),1);
ndx = size(mean_x,1);


invOmega_times_cov_x = invOmega*cov_x;
for i = 1:length(alpha)
    
    invPi = alpha(i)*I + invOmega;
    invPicovx = alpha(i)*cov_x + invOmega_times_cov_x;
       
%     invPicovx = invPi*cov_x;
%     invPicovx = 0.5*(invPicovx + invPicovx');
    
    % compute eq(26) on the grid of alpha values
    obj(i) = logdetns( invPicovx) - trace( invPicovx) +ndx - mean_x'*invPi*mean_x;

end



[obj_alpha,v] = max(0.5*obj);
newAlpha = alpha(v); % choose the maximum
% tt = toc;
% display(sprintf('Mstep_updateAlpha took: %.3g', tt));
% display(sprintf('Optimized alpha: %.3g', newAlpha));
% figure(100);
% semilogx(alpha, 0.5*obj, '--', alpha(v), obj_alpha, 'ro')

% display(newAlpha)

%%

% tic;
% fun = @(a) trace(inv(a*I+ invOmega) - cov_x) - mean_x'*mean_x;  
% alpha0 = 3; 
% alphanew = fzero(fun, alpha0);
% toc;
% 
% display(alphanew)
end
