[a,p] = textread('elapsed_time.txt','%f %f');
for i=1:length(p)
    if (p(i) == -1)
        p(i) = a(i);
    end
end
y = (abs(a-p)./(a)) .* 100;
x = abs(a-p);
[m,i]= max(x);
diff = abs(a(i) - p(i))

count(1) = 0;
count(2) = 0;
count(3) = 0;
count(4) = 0;
count(5) = 0;
t = 1:length(y);
for j=1:length(y)
    if (y(j) < 1.) 
        count(1) = count(1) + 1;
    elseif (y(j) < 5.)
        count(2) = count(2) + 1;
    elseif (y(j) < 10.)
        count(3) = count(3) + 1;
    elseif (y(j) < 20.)
        count(4) = count(4) + 1;
    else 
        count(5) = count(5) + 1;
    end 
end

for j=1:5
    count(j) = 100 .* (count(j)/length(y));
end 
fprintf("relative error (< 1 percent) = %f\n",count(1));
fprintf("relative error (< 5 percent) = %f\n",count(2));
fprintf("relative error (< 10 percent) = %f\n",count(3));
fprintf("relative error (< 20 percent) = %f\n",count(4));
fprintf("relative error (>= 20 percent) = %f\n",count(5));
figure(1);
stem(count);
count(1) = 0;
count(2) = 0;
count(3) = 0;
count(4) = 0;
count(5) = 0;

for j=1:length(y)
    if (x(j) < 5.) 
        count(1) = count(1) + 1;
    elseif (x(j) < 50.)
        count(2) = count(2) + 1;
    elseif (x(j) < 500.)
        count(3) = count(3) + 1;
    elseif (x(j) < 1000.)
        count(4) = count(4) + 1;
    else 
        count(5) = count(5) + 1;
    end 
end

for j=1:5
    count(j) = 100 .* (count(j)/length(y));
end 

fprintf("elapsed time absolute error (< 5 microseconds) = %f\n",count(1));
fprintf("elapsed time absolute error (< 50 microseconds) = %f\n",count(2));
fprintf("elapsed time absolute error (< 500 microseconds) = %f\n",count(3));
fprintf("elapsed time absolute error (< 1 miliseconds) = %f\n",count(4));
fprintf("elapsed time absolute error (>= 1 miliseconds) = %f\n",count(5));
figure(2);
stem(count);
figure(3);
[m,i] = max(x);
fprintf("Max absolute difference %d miliseconds at %d \n", diff/1000, i);
fprintf("Actual = %d miliseconds, Prediction = %d miliseconds \n", a(i)/1000, p(i)/1000);

plot(t,x/1000);
figure(4);
plot(t,a/1000,'b',t,p/1000,'r');


