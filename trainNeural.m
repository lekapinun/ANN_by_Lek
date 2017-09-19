load('xor.mat');
[dataColumn,dataRow] = size(xor);
% question = 'What is your stucture? ' ;
% answer = input(question,'s');
% neural = sscanf(answer,'%u')';
neural = [2,3,2];
%initial empty vaule
y = cell(1,max(neural));
weight = cell(1,max(neural));
bias = cell(1,max(neural));
y{1} = zeros(1,neural(1));
bias{1} = zeros(1,neural(1));
for i = 2:size(neural,2);
    y{i} = zeros(1,neural(i));
    bias{i} = zeros(neural(i),1);
    weight{i} = zeros(neural(i),size(y{i-1},2));    
end
%initial
y{1} = xor(1,1:size(xor,2)-1);
class = xor(1,size(xor,2));
for i = 2:size(weight,2)
    for m = 1:size(weight{i},1)
        for n = 1:size(weight{i},2)
            weight{i}(m,n) = randWeight(size(y{i-1},2));
        end
        bias{i}(m) = randWeight(size(y{i-1},2));
    end
end
%train
for n = 1:1
    for i = 2:size(neural,2)
        for m = 1:neural(i)
            y{i}(m) = logSigmoid(dot(y{i-1},weight{i}(m,:))+bias{i}(m));
        end
    end
end