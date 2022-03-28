function value = multicore()
    pc = parcluster('local')
    parpool(pc, 40)
    n = 2000;
    y = zeros(n,1);
    parfor i = 1:n
        y(i) = max(svd(randn(i)));
    end
end

