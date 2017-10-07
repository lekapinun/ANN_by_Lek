function [data_crossValidation,numClassLabel] = crossValidation(data)
numClassLabel = max(data(:,size(data,2)));
[dataColumn,dataRow] = size(data);
data_sperate = cell(numClassLabel,1);
data_crossValidation = cell(10,1);
for i = 1:dataColumn
   classLabel = data(i,dataRow);
   data_sperate{classLabel} = cat(1,data_sperate{classLabel},data(i,:));
end
for i = 1:numClassLabel
    randIndex = randperm(size(data_sperate{i},1));
    temp = data_sperate{i};
    for j = 1:size(randIndex,1)
        data_sperate{i} = temp(randIndex,:);
    end
end
for i = 1:numClassLabel
    index = 0;
    sperate = round(size(data_sperate{i},1)/10);
    for j = 1:9
        data_crossValidation{j} = cat(1,data_crossValidation{j},data_sperate{i}(index+1:index+sperate,:));
        index = index + sperate;
    end
    data_crossValidation{10} = cat(1,data_crossValidation{10},data_sperate{i}(index+1:size(data_sperate{i},1),:));
end
for i = 1:10
    randIndex = randperm(size(data_crossValidation{i},1));
    temp = data_crossValidation{i};
    for j = 1:size(randIndex,1)
        data_crossValidation{i} = temp(randIndex,:);
    end
end
