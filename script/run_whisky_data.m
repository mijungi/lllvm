

n = 86;
kmat = floor(n./[2, 4, 8, 10, 12, 14, 16, 18, 22, 30, 50]);
% kmat = n./[10, 20, 40];
seedmat = 1:50;

for k_idx = 1:length(kmat)
    k = kmat(k_idx);
    for seed_idx = 1:length(seedmat)
        seed = seedmat(seed_idx);
        
        fprintf(['Testing: k= ' num2str(k) ' and seed = ' num2str(seed)])
        
        test_whiskies_data(seed, k)
        
    end
end
