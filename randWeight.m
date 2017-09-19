function [result] = randWeight(fanin)
rand_weight = rand;
while rand_weight == 0.5
    rand_weight = rand;
end
result = (rand_weight - 0.5) * (1/sqrt(fanin));
