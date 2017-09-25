clear
data_set = struct2cell(load('cross.mat'));
numClassLabel = 2;
data_set = data_set{1};
numInput = size(data_set(1,:),2)-1;
neural = [numInput,3,numClassLabel];
data_crossed = crossValidation(data_set,numClassLabel);
data = cell(10,2);
truthTable = cell(10,2);
correct = cell(10,1);
%create variable
learningRate = 0.1; 
momentumRate = 0.1;
y = cell(1,size(neural,2)); 
weight = y; bias = y; gradient = y;
epoch = 1;
for i = 1:10
    for j = 1:2
        truthTable{i,j} = zeros(numClassLabel,numClassLabel);
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
E = cell(10,1);
collectE = cell(10,1);
collectEpoch = cell(10,1);
indexE = 1;
%initialization
y{1} = zeros(1,neural(1));
bias{1} = y{1}; gradient{1} = y{1};
for i = 2:size(neural,2);
    y{i} = zeros(1,neural(i));
    bias{i} = y{i}; gradient{i} = y{i};
    weight{i} = zeros(neural(i),size(y{i-1},2));  
end
weightOld = weight;
biasOld = bias;
for i = 2:size(weight,2)
    for m = 1:size(weight{i},1)
        for n = 1:size(weight{i},2)
            weight{i}(m,n) = randWeight(size(y{i-1},2));
        end
        bias{i}(m) = randWeight(size(y{i-1},2));
    end
end
tempInit = {y, bias, weight, weightOld, biasOld};
%ANN
for train = 1 : 10
    disp('START')
    disp(train)
    y = tempInit{1};
    bias = tempInit{2};
    weight = tempInit{3};
    weightOld = tempInit{4};
    biasOld = tempInit{5};
    E{train} = zeros(size(data{train,1},1),1);
    epoch = 1;
    %Train
    while epoch <= 1000
        perm = randperm(size(data{train,1},1));
        for n = 1:size(data{train,1},1)
            d = zeros(1,numClassLabel); 
            d(data{train,1}(perm(n),size(data{train,1},2))) = 1;
            y{1} = data{train,1}(perm(n),1:size(data{train,1},2)-1);
            %Calculate output
            for i = 2:size(neural,2)
                for m = 1:neural(i)
                    y{i}(m) = logSigmoid(dot(y{i-1},weight{i}(m,:))+bias{i}(m));
                end
            end
            %Calculate e
            e = d - y{size(neural,2)};
            %Calculate E
            E{train}(n) = 0.5*(e*e');
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
                        weight{i}(l,m) = weight{i}(l,m) + (momentumRate*(weight{i}(l,m)-weightOld{i}(l,m)))+(learningRate*gradient{i}(l)*y{i-1}(m));
                    end
                    bias{i}(l) = bias{i}(l) + (momentumRate*(bias{i}(l)-biasOld{i}(l)))+(learningRate*gradient{i}(l));
                end
            end
            weightOld = temp_weight;
            biasOld = temp_bias;
        end
        collectEpoch{train} = epoch;
        if sum(E{train})/size(E{train},1) < 0.12%0.02
            %disp(sum(E{train})/size(E{train},1))
            disp(epoch)
            break;
        end
        collectE{train}(epoch) = sum(E{train})/size(E{train},1);
        epoch = epoch + 1;
    end
    %Test
    for k = 1:2
        for n = 1:size(data{train,k},1)
             fact = (data{train,k}(n,size(data{train,2},2)));
             y{1} = data{train,k}(n,1:size(data{train,2},2)-1);
             %Calculate output
             for i = 2:size(neural,2)
                 for m = 1:neural(i)
                     y{i}(m) = logSigmoid(dot(y{i-1},weight{i}(m,:))+bias{i}(m));
                 end
             end
             %correct or not
             guess = find(y{size(neural,2)} == max(y{size(neural,2)}));
             truthTable{train,k}(fact,guess) = truthTable{train,k}(fact,guess) + 1;
        end
    end
    correct{train} = zeros(1,2);
    for j = 1:2
        count = 0;
        for k = 1:size(truthTable{train,j},1)
            count = count + truthTable{train,j}(k,k);
        end
        correct{train}(j) = count/sum(sum(truthTable{train,j}));
    end
end

plot(1:size(collectE{1},2),collectE{1},1:size(collectE{2},2),collectE{2},1:size(collectE{3},2),collectE{3},1:size(collectE{4},2),collectE{4},1:size(collectE{5},2),collectE{5},1:size(collectE{6},2),collectE{6},1:size(collectE{7},2),collectE{7},1:size(collectE{8},2),collectE{8},1:size(collectE{9},2),collectE{9},1:size(collectE{10},2),collectE{10})
