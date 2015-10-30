function [ funcs] = funcs_frey_face( )
%FUNCS_FREY_FACE A collection of functions to manipulate frey face data and 
%experimental results.
%
    funcs = struct();
    funcs.show_image_grid = @show_image_grid;
    funcs.batch_resize = @batch_resize;
    funcs.subsample = @subsample;
    funcs.plot_projected_2d = @plot_projected_2d;
    funcs.get_kns_lwb_cube = @get_kns_lwb_cube;
    funcs.run_ksn_lwb_cube = @run_ksn_lwb_cube;

end

function show_image_grid(ff, max_shown)
    % Show face imagesc in a grid. ff is the loaded frey face data of size
    % 560x1965.
    %
    if nargin < 2
        max_shown = 100;
    end
    assert(max_shown > 0);
    figure
    hold on;
    colormap gray;
    max_sqrt = ceil(sqrt(max_shown));
    [d, n] = size(ff);
    for i=1:min(n, max_sqrt^2)
        h = subplot(max_sqrt, max_sqrt, i);
        fi = reshape(ff(:, i), 20, 28)';
        imagesc(fi);
        axis off;
        %set(gca, 'FontSize', 16);
        text(1, 1, sprintf('%d', i), 'Color', 'red', 'Parent', h);
    end
    hold off;
end

function ff_resize = batch_resize(ff, scale)
    % return a new set of imagesc of size d'x n where d' is the scale times the 
    % original dimension of ff (originally 560).
    [d, n] = size(ff);
    f1 = reshape(ff(:, 1), 20, 28);
    f1_resize = imresize(f1, scale);
    ff_resize = zeros(numel(f1_resize), n);
    for i=1:n
        fi = reshape(ff(:, i), 20, 28);
        re = imresize(fi, scale);
        ff_resize(:, i) = re(:);
    end

end

function sub_ff = subsample(ff, sub_n, seed)
    % subsample data to have only sub_n examples using the seed.
    % This function is deterministic given seed.
    oldRng = rng();
    rng(seed);

    [d, n] = size(ff);
    assert(sub_n > 0);
    sub_n = min(n, sub_n);
    I = randperm(n, sub_n);
    sub_ff = ff(:, I);

    rng(oldRng);

end

function ca = plot_projected_2d(mu_x, sub_ff, width_px, G, n_img )
    % Plot the 2d projected points x along with images in sub_ff.
    % - mu_x: dx x n 
    % - sub_ff: dx x n corresponding to mu_x
    % - width_px: the width in pixels of each image in sub_ff
    % - G: n x n graph
    % - n_img: number of images to show
    %
    [dx, n] = size(mu_x);
    mu_x = 150*mu_x;
    assert(n == size(sub_ff, 2), 'size of mu_x and sub_ff do not match');
    assert(isscalar(n_img));
    assert(n_img > 0);
    d = size(sub_ff, 1);

    n_img = min(n, n_img);
    I = randperm(n, n_img);
    figure 
    hold on
    plot(mu_x(1, :), mu_x(2, :), 'o');
    ca = gca;
    h_px = d/width_px;
    for i=1:n_img
        im_i = reshape(sub_ff(:, I(i)), width_px, [])';
        imagesc(mu_x(1, I(i)), mu_x(2, I(i)), flipud(im_i) );
        colormap gray;
    end
    hold off

end

function ksn_lwb = get_ksn_lwb_cube(ks, ns, seeds)
    % scan all files in saved/ex1 and construct a 3d array of lower bounds
    % of size #k x #seeds x #n 
    % - ks, ns, seeds are array of numbers 
    
    fg = funcs_global();
    %files = fg.expSavedFolder(1);
    ksn_lwb = zeros(length(ks), length(seeds), length(ns));
    for ki=1:length(ks)
        k = ks(ki);
        for si=1:length(seeds)
            s = seeds(si);
            for ni=1:length(ns)
                n = ns(ni);
                fname = sprintf('frey-s%d-k%d-n%d.mat', s, k, n);
                fpath = fg.expSavedFile(1, fname);
                display(sprintf('loading %s', fpath));
                loaded = load(fpath, 'results');
                % Take the lower bound value in the last iteration.
                ksn_lwb(ki, si, ni) = loaded.results.lwbs(end);
            end
        end
    end

end 

function ksn_lwb = run_ksn_lwb_cube()
    ks = [ 2, 3, 5, 10, 20, 40, 60, 80, 100];
    seeds = 1:5;
    ns = 500;
    ksn_lwb = get_ksn_lwb_cube(ks, ns, seeds);
    % ks x seeds 
    [max_lwb, maxi] = max(ksn_lwb(:));
    [bki, bsi] = ind2sub(size(ksn_lwb), maxi);

    timeStamp = clock();
    fg = funcs_global();
    fname = sprintf('lwb_cube.mat' );
    savePath = fg.expSavedFile(1, fname);
    save(savePath);

    % export all variables to the base workspace.
    allvars = who;
    warning('off','putvar:overwrite');
    putvar(allvars{:});

end

