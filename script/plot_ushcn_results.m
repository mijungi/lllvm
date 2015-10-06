function [ ] = plot_ushcn_results(seed )
%PLOT_USHCN_RESULTS Plot projected 2d results of the USHCN climate data.
%

% tmmpr. n=609. seed: 3, 30, 33, 61, 119, 136
%%
oldRng = rng();
rng(1);

fg = funcs_global();
ex = 2;
% min/max temperatures and precipitation from 2005-2014. Data normalized
% (standardized).
%dataName = 'tmmpr_f05t14_nor';
dataName = 'prcp_10y_t14';
%seed = 30;
k = 50;

%seed = 1;
%k = 7;
n = 609;
fname = sprintf('ushcn-d%s-s%d-k%d-n%d.mat', dataName, seed, k, n);
fpath = fg.expSavedFile(ex, fname);

%% experimental results
display(sprintf('loading result: %s', fpath));
exr = load(fpath);

funcs = funcs_ushcn();
% make 2xn 
mu_x = reshape(exr.results.mean_x, 2, []);

%G_mux = makeKnnG(mu_x, k);
%n = size(mu_x, 2);

lalo_pivot = [48.9775; -122.7928];
bottom_mid_pivot = [27.14; -98.12];
% center
%lalo_pivot = [40.; -95];
% bottom right
%lalo_pivot = [24.55; -81.75];
lola_pivot = [lalo_pivot(2); lalo_pivot(1)];
% distances to the pivot point
subSampleInd = exr.subSampleInd;
lola = [exr.data.laloel(2, subSampleInd); exr.data.laloel(1, subSampleInd)];
dist2pivot = sqrt(sum(bsxfun(@minus, lola, lola_pivot).^2, 1));
% sizes of the points.
S = sqrt(sum(bsxfun(@minus, lola(2, :), bottom_mid_pivot(2, :)).^2, 1));
dist2sizePivot = 1.1.^S/3000 ;
%dist2sizePivot = 1.09.^S/3000 ;

% plot the true station locations 
fig = funcs.plot_stations_2d(exr.data, subSampleInd, lalo_pivot, dist2sizePivot);

gplvm_proj = exr.gplvm_proj;
isomap_k = k;
%
[ltsa_x ] = ltsa(exr.Y', 2, k);

[isomap_x, isomap_mapping] = isomap(exr.Y', 2, isomap_k);
ltsa_k = isomap_k;

lle_k = isomap_k;
[lle_x, lle_mapping] = lle(exr.Y', 2, lle_k);
% plot LL-LVM's projected points
figure;
hold on;
scatter(mu_x(1, :), mu_x(2, :), dist2sizePivot, dist2pivot, 'fill');
set(gca, 'FontSize', 16);
axis off 
title(sprintf('LLLVM. k=%d. %d weather stations', k, n));
colormap jet;
%grid on;
hold off;

%% plot gplvm results
%%gplvm_proj = gplvm(ex.Y', 2)';
figure;
hold on;
scatter(gplvm_proj(1, :), gplvm_proj(2, :), dist2sizePivot, dist2pivot, 'fill');
set(gca, 'FontSize', 16);
axis off 
title(sprintf('GPLVM. %d weather stations', n));
colormap jet;
%grid on;
hold off;

% plot isomap results.
% Isomap will perform dimensionality reduction on the largest connected component. 
% Need to make sure that all points are connected. 
figure;
hold on;
scatter(isomap_x(:, 1), isomap_x(:, 2), dist2sizePivot(isomap_mapping.conn_comp), ...
    dist2pivot(isomap_mapping.conn_comp), 'fill');
set(gca, 'FontSize', 16);
axis off 
title(sprintf('Isomap. k=%d.', isomap_k));
colormap jet;
%grid on;
hold off;

% plot ltsa results
figure;
hold on;
scatter(ltsa_x(:, 1), ltsa_x(:, 2), dist2sizePivot,   dist2pivot, 'fill');
set(gca, 'FontSize', 16);
axis off 
title(sprintf('LTSA. k=%d.', ltsa_k));
colormap jet;
%grid on;
hold off;


% plot ltsa results
figure;
hold on;
scatter(lle_x(:, 1), lle_x(:, 2), dist2sizePivot(lle_mapping.conn_comp), ...
    dist2pivot(lle_mapping.conn_comp), 'fill');
set(gca, 'FontSize', 16);
axis off 
title(sprintf('LLE. k=%d.', lle_k));
colormap jet;
%grid on;
hold off;

%% Compare shortest path distances. 
% Define Dtrue as the true graph distance computed by running Dijkstra on the k-NN graph 
% of the actual coordinates.
% For each result, compute a graph GE (G estimated) from the projected points, compute 
% a k-NN graph, run Dijkstra to get GED, and compare the GED with Dtrue entrywise.
%
%display(' # geodesic errors');
%Gtrue = makeKnnG(lola, k);
%allIndices = 1:size(lola, 2);
%Dtrue = dijkstra(Gtrue, allIndices);

%%%
%% lllvm 
%G_lllvm = makeKnnG(mu_x, k);
%D_lllvm = dijkstra(G_lllvm, allIndices);
%err_lllvm = mean(abs(Dtrue(:) - D_lllvm(:)));

%G_isomap = makeKnnG(isomap_x', k);
%D_isomap = dijkstra(G_isomap, allIndices);
%err_isomap = mean(abs(Dtrue(:) - D_isomap(:)));

%%%
%G_gplvm = makeKnnG(gplvm_proj, k);
%D_gplvm = dijkstra(G_gplvm, allIndices);
%err_gplvm = mean(abs(Dtrue(:) - D_gplvm(:)));

%G_ltsa = makeKnnG(ltsa_x', k);
%D_ltsa = dijkstra(G_ltsa, allIndices);
%err_ltsa = mean(abs(Dtrue(:) - D_ltsa(:)));

%G_lle = makeKnnG(lle_x', k);
%D_lle = dijkstra(G_lle, allIndices);
%err_lle = mean(abs(Dtrue(:) - D_lle(:)));

%display(sprintf('Errors compared to the true geodesics.'));
%display(sprintf('LL-LVM geodesic error: %.3f', err_lllvm));
%display(sprintf('GP-LVM geodesic error: %.3f', err_gplvm));
%display(sprintf('ISOMAP geodesic error: %.3f', err_isomap));
%display(sprintf('LTSA geodesic error: %.3f', err_ltsa));
%display(sprintf('LLE geodesic error: %.3f', err_lle));

fg = funcs_global();
fpath = fg.scriptSavedFile(sprintf('plot_ushcn-d%s-s%d-k%d.mat', dataName, seed, k));

rng(oldRng);

clear fg  oldRng funcs
save(fpath);
%%%%%
end


