function [result] = calculateGradientHidden(w,g,y)
result = diffLogSigmoid(y)*dot(w,g);