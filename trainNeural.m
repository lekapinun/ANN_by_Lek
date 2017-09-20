load('xor.mat');
[dataColumn,dataRow] = size(xor);
% question = 'What is your stucture? ' ;
% answer = input(question,'s');
% neural = sscanf(answer,'%u')';
neural = [2,3,2];
%initial empty vaule
y = cell(1,max(neural));
% weight = cell(1,max(neural));
% bias = cell(1,max(neural));
y{1} = zeros(1,neural(1));
% bias{1} = zeros(1,neural(1));
gradient{1} = zeros(neural(1),1);
E = zeros(size(xor,1),1);
learningRate = 0.05;
momemtumRate = 0.05;
for i = 2:size(neural,2);
    y{i} = zeros(1,neural(i));
%     bias{i} = zeros(neural(i),1);
%     weight{i} = zeros(neural(i),size(y{i-1},2));  
    gradient{i} = zeros(neural(i),1);
end
weightOld = weight;
biasOld = bias;
%initial
y{1} = xor(1,1:size(xor,2)-1);
% class = xor(1,size(xor,2));
% class = zeros(1,2);
% class(1,xor(1,size(xor,2))) = 1;
for i = 2:size(weight,2)
    for m = 1:size(weight{i},1)
        for n = 1:size(weight{i},2)
%             weight{i}(m,n) = randWeight(size(y{i-1},2));
        end
%         bias{i}(m) = randWeight(size(y{i-1},2));
    end
end
%Train
for x = 1:1000
for n = 1:160
    d = zeros(1,2);
    d(xor(n,size(xor,2)))  = 1;
    %Calculate output
    for i = 2:size(neural,2)
        for m = 1:neural(i)
            y{i}(m) = logSigmoid(dot(y{i-1},weight{i}(m,:))+bias{i}(m));
        end
    end
    %Calculate e
    e = d - y{3};
    %Calculate E
    E(n) = 0.5*(e*e');
    %Calculate gradient output
    for i = 1:neural(end)
        gradient{size(neural,2)}(i) = calculateGradientOutput(e(i),y{size(neural,2)}(i));
    end
    %Calculate gradient hidden
    for i = size(neural,2)-1:-1:2
        for j = 1:neural(i)
            gradient{i}(j) = calculateGradientHidden(weight{i+1}(:,j),gradient{i+1},y{i}(j)) ;
        end
    end
    %Change weight
    temp_weight = weight;
    temp_bias = bias;
    for i = size(neural,2):-1:2
        for l = 1:size(weight{i},1)
            for m = 1:size(weight{i},2)
                weight{i}(l,m) = weight{i}(l,m) + (momemtumRate*(weight{i}(l,m)-weightOld{i}(l,m)))+(learningRate*gradient{i}(l)*y{i-1}(m));
            end
            bias{i}(l) = bias{i}(l) + (momemtumRate*(bias{i}(l)-biasOld{i}(l)))+(learningRate*gradient{i}(l));
        end
    end
    weightOld = temp_weight;
    biasOld = temp_bias;
end
end
