net = perceptron();

load('data2d.mat');
P=data2d([1,2],:);
T=data2d(3,:);

%load('data2dns.mat');
%P=data2dns([1,2],:);
%T=data2dns(3,:);

for col= 1:length(T)
    if( T(:,col) == -1)
        T(:,col) = 0;
    end  
end

net = train(net,P,T);
sectionA(net,T,P)



function sectionA(net,T,P)
w = net.iw{1,1};
b = net.b{1};

a = net(P);

x = P(1,:);
y = P(2,:);

gscatter(x,y,T); hold on;
xlabel('first row');
ylabel('second row');
plot(x, x.*w+b);
hold off;
end


function sectionB()

end