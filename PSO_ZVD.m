clear
%lệnh mô phỏng neural network trên simulink
%gensim(tên network)

%% Problem Definition
warning('off');
%m=0.536; m1=0.375; m2=0.5;
m =5; ml =2;m1=2; m2=0.5;
g =9.81; L2=0.1;      
%waof = 0.5*(sqrt(9.81/0.6)+sqrt(9.81/0.2)); waol = sqrt(9.81/0.4);
n1 = 5;  n2 = 1;                 %trọng số hàm mục tiêu
nVar = 2; 
l_min = 0.1; l_step = 0.02; l_max = 1;

for L1 = l_min:l_step:l_max
%L1 = 0.05; 
mc = 1;
D1=mc/(m1+mc);
  
L=L1+L2; 
l_ini=L;

%Bo input shaping
alpha = (1/L1 + 1/L2)*(m1+m2)/m1;
beta = (((m1+m2)/m1)^2)*((1/L1 + 1/L2)^2) - 4*(m1+m2)/(m1*L1*L2);
w1 = sqrt(g*(alpha - sqrt(beta))/2);
w2 = sqrt(g*(alpha + sqrt(beta))/2);

damping_ratio = 0;
wd1 = w1*sqrt(1-damping_ratio^2);
wd2 = w2*sqrt(1-damping_ratio^2);

%VarMin = [0,2*pi/(3*sqrt(g/l)),4*pi/(3*sqrt(g/l))];                 %không gian tìm kiếm
VarMin = [0,pi/(wd1)];
VarMax = [pi/(wd1),2*pi/(wd1)];
%MaxIt=37;                                   %lần lặp tối đa
MaxIt=20;  
VarSize = [1 nVar];  
VelMax = 0.1*(VarMax-VarMin);
VelMin = -VelMax;

%% PSO parameter
it=1;
%w =0.9;                        %khối lượng quán tính
wmax=0.9; % inertia weight
wmin=0.4; % inertia weight
c1 = 2;                           %trọng số personal
c2 = 2;                           %trọng số global
nPop = 10;                         %dân số


%% Initialization
empty_particle.Position = [];
empty_particle.Velocity = [];
empty_particle.Best.Position = [];
empty_particle.Best.J = [];
particle = repmat(empty_particle, nPop, 1);
GlobalBest.J = inf;
for i = 1:nPop
    %khởi tạo vị trí ban đầu
    %particle(i).Position = rand(VarSize).*(VarMax - VarMin) + VarMin;
    particle(i).Position =unifrnd(VarMin, VarMax, VarSize);
    %khởi tạo vận tốc ban đầu
    particle(i).Velocity = zeros(VarSize);

    %khởi tạo hàm mục tiêu
    particle(i).J = inf;

    %cập nhập pbest
    particle(i).Best.Position = particle(i).Position;
    particle(i).Best.J = particle(i).J;

    % cập nhật gbest
    if particle(i).Best.J<GlobalBest.J
        GlobalBest.J = particle(i).Best.J;
    end
end
BestJ = zeros(MaxIt, 1);

%% PSO Loop

for it = 1:MaxIt
   
    w=wmax-(wmax-wmin)*it/MaxIt; % update inertial weight
    for i = 1:nPop
        simout = sim('Model1_3');

%---------------- tính giá trị hàm mục tiêu --------------------
           particle(i).J = n1*simout.IAE.Data(end)+n2*simout.MAX1.Data(end)+n2*simout.MAX2.Data(end);  
        %Fit=0;
        %for a=1:length(simout.Fit.Data)
        %    if simout.Fit.Data(a)>Fit
        %    Fit=simout.Fit.Data(a);
        %    end
        %end
        %particle(i).J=Fit;
          % cập nhật pbest
        if particle(i).J < particle(i).Best.J

            particle(i).Best.Position = particle(i).Position;
            particle(i).Best.J = particle(i).J;

            % cập nhật gbest
            if particle(i).Best.J<GlobalBest.J

                GlobalBest = particle(i).Best;

            end
        end

        % cập nhật vận tốc
        particle(i).Velocity = w*particle(i).Velocity ...
            +c1*rand(VarSize).*(particle(i).Best.Position-particle(i).Position) ...
            +c2*rand(VarSize).*(GlobalBest.Position-particle(i).Position);

        % điều chỉnh vận tốc
        particle(i).Velocity = max(particle(i).Velocity, VelMin);
        particle(i).Velocity = min(particle(i).Velocity, VelMax);

        % cập nhật vị trí
        particle(i).Position = particle(i).Position + particle(i).Velocity;

        % mirror vận tốc
        IsOutside = (particle(i).Position<VarMin | particle(i).Position>VarMax);
        particle(i).Velocity(IsOutside) = -particle(i).Velocity(IsOutside);

        % cập nhật vị trí
        particle(i).Position = max(particle(i).Position, VarMin);
        particle(i).Position = min(particle(i).Position, VarMax);

    end
    %display giá trị mỗi lần lặp
    BestJ(it) = GlobalBest.J;
%      disp(['Iter ' num2str(it) ': Best J = ' num2str(BestJ(it)) ' || Best Position: ' num2str(GlobalBest.Position) ]);
     
end

  disp(['  *l1 = ' num2str(L1)]);
  disp(['mc = ' num2str(mc)]);  
  disp('GlobalBest.Position=');
  disp([num2str(GlobalBest.Position(1,1)) '  ' num2str(GlobalBest.Position(1,2)) '  ' num2str(GlobalBest.Position(1,3))]);
  disp('GlobalBest.J=');
  disp(num2str(GlobalBest.J));
%     disp(['time = ' num2str(time)]);
 end
disp('Fisnish');
%% Results
% figure;
% plot(BestJ, 'LineWidth', 1.5);
% xlabel('Iteration');
% ylabel('Best J');
% grid on;