function [result] = calculateGradientOutput(e,y)
result = e*diffLogSigmoid(y);