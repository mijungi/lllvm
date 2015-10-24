
% n = 3000;
% kmat = 200;
% seedmat = 1;
% dxmat = 2;
% n = 1000;
n = 400; % 80 times 5
kmat = floor(n./[8, 6, 4, 2, 1]);
% kmat = n./[8, 4, 2];
% kmat = n/8;
% kmat = n/4;
% kmat = n/2;
seedmat = 1:10;

% dxmat = [2, 4, 8];
% dxmat = 2;

dx = 2;

% for dx_idx = 1: length(dxmat)
%     dx = dxmat(dx_idx);
for k_idx = 1:length(kmat)
    k = kmat(k_idx);
    for seed_idx = 1:length(seedmat)
        seed = seedmat(seed_idx);
        fprintf(['USPS_k=' num2str(k) '_dx=' num2str(dx) '_s=' num2str(seed)])
        test_USPS_handwritten_digit_data(seed, k, dx);
    end
end
% end
