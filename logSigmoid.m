function [result] = logSigmoid(number)
result = 1/(1+exp(-number));