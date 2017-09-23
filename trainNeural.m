clear
data_set = struct2cell(load('xor2.mat'));
data_set = data_set{1};
data_crossed = crossValidation(data_set,2);
data = cell(10,2);
truthTable = cell(2,2);
neural = [2,3,2];
numClassLabel = 2;
%create variable
y = cell(1,size(neural,2)); 
weight = y; bias = y; gradient = y;
y{1} = zeros(1,neural(1));
bias{1} = y{1}; gradient{1} = y{1};
learningRate = 0.1; 
momemtumRate = 0.1;
for i = 2:size(neural,2);
    y{i} = zeros(1,neural(i));
    bias{i} = y{i}; gradient{i} = y{i};
    weight{i} = zeros(neural(i),size(y{i-1},2));  
end
weightOld = weight;
biasOld = bias;
epoch = 1;
for i = 1:10
    for j = 1:2
        truthTable{i,j} = zeros(numClassLabel,numClassLabel);
    end
end
%initial
for i = 2:size(weight,2)
    for m = 1:size(weight{i},1)
        for n = 1:size(weight{i},2)
            weight{i}(m,n) = randWeight(size(y{i-1},2));
        end
        bias{i}(m) = randWeight(size(y{i-1},2));
    end
end
for i = 1:10
    for j = 1:10
        if j ~= i
            data{i,1} = cat(1,data{i,1},data_crossed{j});
        else
            data{i,2} = cat(1,data{i,2},data_crossed{j});
        end
    end
end
numTrain = 0;
for i = 1:10
    numTrain = numTrain + size(data{i},1);
end
E = zeros(numTrain,1);
xxx = 1;
%Train
while epoch < 1000
    indexE = 1;
    for train = 1 : 10
        for n = 1:size(data{train,1},1)
            d = zeros(1,2); 
            d(data{train,1}(n,size(data{train,1},2))) = 1;
            y{1} = data{train,1}(n,1:size(data{train,1},2)-1);
            %Calculate output
            for i = 2:size(neural,2)
                for m = 1:neural(i)
                    y{i}(m) = logSigmoid(dot(y{i-1},weight{i}(m,:))+bias{i}(m));
                end
            end
            %Calculate e
            e = d - y{size(neural,2)};
            %Calculate E
            E(indexE) = 0.5*(e*e');
            indexE = indexE + 1;
            %Calculate gradient output
            for i = 1:neural(end)
                gradient{size(neural,2)}(i) = e(i)*diffLogSigmoid(y{size(neural,2)}(i));
            end
            %Calculate gradient hidden
            for i = size(neural,2)-1:-1:2
                for j = 1:neural(i)
                    gradient{i}(j) = diffLogSigmoid(y{i}(j))*dot(weight{i+1}(:,j),gradient{i+1});
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
    if sum(E)/size(E,1) <= 0.005
        break;
    end
    epoch = epoch + 1; 
end
%save value to truth table
for train = 1 : 10
    for k = 1:2
        dataCheck = data{train,k};
        for l = 1:size(dataCheck,1)
            fact = dataCheck(l,size(dataCheck,2));
            y{1} = dataCheck(l,1:size(dataCheck,2)-1);
            %Calculate output
            for i = 2:size(neural,2)
                for m = 1:neural(i)
                    y{i}(m) = logSigmoid(dot(y{i-1},weight{i}(m,:))+bias{i}(m));
                end
            end
            guess = find(y{size(neural,2)} == max(y{size(neural,2)}));
            truthTable{train,k}(fact,guess) = truthTable{train,k}(fact,guess) + 1;
        end
    end
end
