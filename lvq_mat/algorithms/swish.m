function act = swish(x,beta)
  act = x .* (1./(1. + exp(-beta.*x)));
end 