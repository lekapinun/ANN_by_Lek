% function [dataResult] = crossValidation(data,numClassLabel)
load('xor.mat');
data = xor;
numClassLabel = 2;
[dataColumn,dataRow] = size(data);
% disp(data(:,dataRow));
count = zeros(1,numClassLabel);
sperate = floor(dataColumn/numClassLabel);
surplus = dataColumn-(sperate*(numClassLabel-1));
data_sperate = zeros(sperate,dataRow,numClassLabel-1);
data_surplus = zeros(surplus,dataRow);
for i = 1:dataColumn
   check = data(i,3);
   count(1,check) = count(1,check) + 1;
   if check < numClassLabel
       data_sperate(count(1,check),:,check) = data(i,:);
   else
       data_surplus(count(1,check),:) = data(i,:);
   end
end
[~,~,size_data_sperate] = size(data_sperate);
data_1_9 = zeros(1,3,9);
[size_sperate,~] = size(data_sperate(:,:,1));
tenth_sperate = floor(size_sperate/10);
for i = 1:9
    index = ((i-1)*tenth_sperate)+1;
    data_1_9(1:tenth_sperate,1:dataRow,i) = data_sperate(index:index+tenth_sperate-1,1:dataRow,1);
end
data_10 = data_sperate(size_sperate-tenth_sperate+1:size_sperate,1:dataRow,1);
if size_data_sperate > 1
    for i = 2:size_data_sperate
        [size_sperate,~] = size(data_sperate(:,:,i));
        tenth_sperate = floor(size_sperate/10);
        temp_size = size(data_1_9,1);
        for j = 1:9
            index = ((j-1)*tenth_sperate)+1;
            data_1_9(temp_size+1:temp_size+tenth_sperate,:,j) = data_sperate(index:index+tenth_sperate-1,1:dataRow,i);
%             data_1_9 = cat(3,data_1_9,data_sperate(index:index+tenth_sperate-1,1:dataRow,i));
        end
        data_10 = cat(3,data_10,data_sperate(size_sperate-tenth_sperate+1:size_sperate,1:dataRow,i));
    end
end
[size_surplus,~] = size(data_surplus(:,:));
tenth_surplus = floor(size_surplus/10);
temp_size = size(data_1_9,1);
for j = 1:9
    index = ((j-1)*tenth_surplus)+1;
    data_1_9(temp_size+1:temp_size+tenth_surplus,:,j) = data_surplus(index:index+tenth_sperate-1,1:dataRow);
end
data_10(size(data_10,1)+1:size(data_10,1)+(size_sperate-(tenth_surplus*9)),:) = data_surplus(size_sperate-tenth_sperate+1:size_sperate,1:dataRow);
dataResult = [];
for i = 1:9
    dataResult = cat(1,dataResult,data_1_9(:,:,i));
end
dataResult = cat(1,dataResult,data_10);





% for i = 1:numClassLabel-1
%     [m,n] = size(data_sperate(:,:,i));
%     k = 1;
%     for j = 1:m
%         data_10()
%     end
% end

% for i = 1:numClassLabel-1
%     random_index = randperm(sperate);
%     temp = data_sperate(:,:,i);
%     for j = 1:sperate
%         data_sperate(j,:,i) = temp(random_index(j),:);
%     end
% end
% random_index = randperm(surplus);
% temp = data_surplus(:,:,i);
% for j = 1:surplus
%     data_surplus(j,:) = temp(random_index(j),:);
% end
% k = 1;
% while k <= dataColumn
%     
% end
% for k = 1:dataColumn
%     
% end


% dataResult = zeros(dataColumn,dataRow);
