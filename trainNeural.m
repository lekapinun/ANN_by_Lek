clear
load('xor2.mat');
[dataColumn,dataRow] = size(xor);
% question = 'What is your stucture? ' ;
% answer = input(question,'s');
% neural = sscanf(answer,'%u')';
neural = [2,3,3,2];
%initial empty vaule
y = cell(1,size(neural,2));
weight = cell(1,size(neural,2));
bias = cell(1,size(neural,2));
gradient = cell(1,size(neural,2));
y{1} = zeros(1,neural(1));
bias{1} = zeros(1,neural(1));
gradient{1} = zeros(neural(1),1);
E = zeros(size(xor,1),1);
learningRate = 0.1;
momemtumRate = 0.1;
for i = 2:size(neural,2);
    y{i} = zeros(1,neural(i));
    bias{i} = zeros(neural(i),1);
    weight{i} = zeros(neural(i),size(y{i-1},2));  
    gradient{i} = zeros(neural(i),1);
end
weightOld = weight;
biasOld = bias;
epoch = 1;
%initial
y{1} = xor(1,1:size(xor,2)-1);
for i = 2:size(weight,2)
    for m = 1:size(weight{i},1)
        for n = 1:size(weight{i},2)
            weight{i}(m,n) = randWeight(size(y{i-1},2));
        end
        bias{i}(m) = randWeight(size(y{i-1},2));
    end
end
%Train
while epoch < 1000
    for n = 1:160
        d = zeros(1,2);
        d(xor(n,size(xor,2)))  = 1;
        y{1} = xor(n,1:size(xor,2)-1);
        %Calculate output
        for i = 2:size(neural,2)
            for m = 1:neural(i)
                y{i}(m) = logSigmoid(dot(y{i-1},weight{i}(m,:))+bias{i}(m));
            end
        end
        %Calculate e
        e = d - y{size(neural,2)};
        %Calculate E
        E(n) = 0.5*(e*e');
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
    if sum(E)/size(E,1) <= 0.005
        break;
    end
    epoch = epoch + 1; 
end
correct = zeros(160,1);
for a = 1:160
   d = zeros(1,2);
   d(xor(a,size(xor,2))) = 1;
   y{1} = xor(a,1:size(xor,2)-1);
   %Calculate output
   for i = 2:size(neural,2)
      for m = 1:neural(i)
         y{i}(m) = logSigmoid(dot(y{i-1},weight{i}(m,:))+bias{i}(m));
      end
   end
   y{4}(y{4} == max(y{4})) = 1;
   y{4}(y{4} ~= max(y{4})) = 0;
   if d == y{4}
       correct(a) = 1;
   end
end
