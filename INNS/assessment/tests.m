% for col = 1:3
%     plot(N,t_perf(:,col));
%     hold on
% end
% hold off
% legend("traincgb","traincgf","traincgp");
u_thresh = min(test_CE) + 0.01;
l_thresh = min(test_CE) - 0.01;

find(test_CE > l_thresh & test_CE < u_thresh)

