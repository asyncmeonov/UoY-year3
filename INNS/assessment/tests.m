% for col = 1:3
%     plot(N,t_perf(:,col));
%     hold on
% end
% hold off
% legend("traincgb","traincgf","traincgp");
thresh = min(test_CE) + 0.01;
min(find(test_CE < thresh))

