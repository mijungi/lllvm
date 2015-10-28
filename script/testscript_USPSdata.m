
clear all;
clc;

n = 400; % 80 times 5
% kmat = floor(n./[8, 6, 4, 2]);

% kmat = floor(n/100); % 
% kmat = floor(n/80); % running on Zoltan's machine
kmat = floor(n/50); % running on Zoltan's machine
% kmat = floor(n/40); % running on Zoltan's machine 
% kmat = floor(n/30);
% kmat = floor(n/20); % screen -r 20892 
% kmat = floor(n/15); % running on Zoltan's machine
% kmat = floor(n/10); % screen -r 5636
% kmat = floor(n/8); % screen -r 5758
% kmat = floor(n/6); % screen -r 5990
% kmat = floor(n/4); % screen -r 15608
% kmat = floor(n/2); % screen -r 15919

% kmat = floor(n./[100, 80, 50, 40, 30, 20, 15, 10, 8, 6, 4, 2]);

seedmat = 6:10;

dx = 2;

for k_idx = 1:length(kmat)
    k = kmat(k_idx);
    for seed_idx = 1:length(seedmat)
        seed = seedmat(seed_idx);
        fprintf(['USPS_k=' num2str(k) '_dx=' num2str(dx) '_s=' num2str(seed)])
        test_USPS_handwritten_digit_data(seed, k, dx);
    end
end

