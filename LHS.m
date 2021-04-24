n = 70;
n_c = 4;
X = zeros(n, n_c);

% day night
lcl = [50,0.0001,2,2];
ucl = [400, 0.01, 21, 11];

X(:, 1:n_c) = lhsdesign(n, n_c, 'criterion', 'maximin');
for i = 1:n_c
    X(:, i) = X(:,i)*abs(ucl(1,i)-lcl(1,i))+lcl(1,i);
end
csvwrite('LHS-mnist.csv', X)