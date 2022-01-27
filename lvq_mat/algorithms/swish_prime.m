function act = swish_prime(x,beta)

sgd = (1./(1. + exp(-beta .* x)));
act = beta .* x .* sgd + sgd .* (1. - beta .* x .* sgd);
end 
