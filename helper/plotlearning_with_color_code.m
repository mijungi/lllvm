function plotlearning_with_color_code(m,n,N,C_Mu,Y, col)
% [N,m,n,Y,~,~,dmat] = fixedparams;

% global p_idx

% figure; 
plotY = reshape(Y,n,N);
% scatter3( plotY(1,:) , plotY(2,:) , plotY(3,:), '.');
scatter3( plotY(1,:) , plotY(2,:) , plotY(3,:) , [] , col , '.');
% title('Manifold (fill circle); inferred manifold (plus); inferred tangent space (plane)'); hold on;

% % Plot learned data
% Y_l = sum( (repmat(C_Mu,N,1)*repmat(X_Mu,1,N)) .* kron(eye(N),ones(n,1)) , 2);
% plotY_l = reshape(Y_l,n,N);
% scatter3( plotY_l(1,:) , plotY_l(2,:) , plotY_l(3,:) , [] , col , '+');

% Plot sampled tangent spaces if 2D -> 3D
hold on; 
if (n == 3) && (m == 2)
%     p_N = 160; % plot this many tangent spaces
    p_N = N;
    %e   = 0.10*(max(reshape(plotY(1:2,:),1,2*N))-min(reshape(plotY(1:2,:),1,2*N)));
    e = 0.1*(max(max(plotY))- min(min(plotY)));
    %mean(reshape(dmat,1,N*N)); % plot tangent space around this plane
    E   = [ -1 -1 -1  0  0  0  1  1  1; ...
            -1  0  1 -1  0  1 -1  0  1] ;
%     if p_idx == 0
%         p_idx = randperm(N); 
%         p_idx = sort(p_idx(1:p_N));
%     end

% mijung changed p_idx = 1:p_N;

    p_idx = 1:p_N;
    
    for ii = 1:numel(p_idx)
        pp = p_idx(ii);
        tangent = C_Mu(:, (pp-1)*m+1:m*pp)*E;
        tangent = e * tangent /norm(tangent); % normalize
%         tangent = e * tangent / max(max(tangent)); % normalize
        tspace  = repmat(plotY(:,pp),1,9) + tangent;
%         mesh( reshape(tspace(1,:),3,3), reshape(tspace(2,:),3,3), reshape(tspace(3,:),3,3), ones(3,3));
        mesh( reshape(tspace(1,:),3,3), reshape(tspace(2,:),3,3), reshape(tspace(3,:),3,3), col(ii)*ones(3,3));
        grid off; 
%         axis off;
        hold on; 
%         if ii==numel(p_idx)
%             scatter3( plotY(1,ii) , plotY(2,ii) , plotY(3,ii));
%         end
    end
end

% axis([-30000 30000 -40000 120000 -250000 150000])

hold off; 
% end % \plotlearning