rows = 1000;
cols = 1500;

%% Ones Function
X = ones(rows,cols);

%% Loop Assignment Without Preallocation
for r = 1:rows
    for c = 1:cols
        X(r,c) = 1;
    end
end

%% Loop Assignment With Preallocation
X = zeros(rows,cols);
for r = 1:rows
    for c = 1:cols
        X(r,c) = 1;
    end
end
