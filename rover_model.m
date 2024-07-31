clc, clear, close all;
addpath casadi-3.6.4-windows64-matlab2018b\
addpath rbf\
import casadi.*

M = 600;
W = M*9.81;
Iz = 1000;
lf = 1.3;
lr = 1.2;
tf = 1.3;
tr = 1.2;

reff = 0.35;
rl = 0.3;
Iww = 1;

L = lf+lr;

h = 0.25;
eps = 1e-4;

fs = 50; % Sampling frequency
Ts = 1/fs;
N = 20;

ns = 7;
nu = 5;

Q = 250*diag([1, 1, 1, 0.3, 0.1, 0.1, 0.1]);
QN = Q;
R = diag([ones(1, nu)]);

QCell = repmat({Q}, 1, N);
BigQ = spblkdiag(QCell{:});

RCell = repmat({R}, 1, N);
BigR = spblkdiag(RCell{:});

y = MX.sym('y', ns);
u = MX.sym('u', nu);

% Y
% 1. X
% 2. Y
% 3. psi
% 4. Vx
% 5. Vy
% 6. Dpsi
% 7. delta

% U
% 1. Ddelta
% 2. kappa_fl
% 3. kappa_fr
% 4. kappa_rl
% 5. kappa_rr

% Velocity vector in chassis frame
v_chassis = [y(4); y(5); 0];

% Wheel contact point position vector in chassi frame
r_fl = [lf; tf/2; 0];
r_fr = [lf; -tf/2; 0];
r_rl = [-lr; tf/2; 0];
r_rr = [-lr; -tf/2; 0];

% Chassis angular yaw velocity vector

Dpsi = [0; 0; y(6)];

% Calculate angular velocity components for wheel contact point
v_ang_fl = cross(Dpsi, r_fl);
v_ang_fr = cross(Dpsi, r_fr);
v_ang_rl = cross(Dpsi, r_rl);
v_ang_rr = cross(Dpsi, r_rr);

% Velocity at the tire contact point in chassis frame
v_fl = v_chassis + v_ang_fl;
v_fr = v_chassis + v_ang_fr;
v_rl = v_chassis + v_ang_rl;
v_rr = v_chassis + v_ang_rr;

% tranform front wheels frame due to steering rotation (local to global)
Rz = [cos(y(7)) -sin(y(7)) 0;
      sin(y(7)) cos(y(7))  0;
      0            0       1];

Rzp = Rz.'; % global to local (because rotaiton matrix is orthogonal Rz^-1 = Rz.')

v_fl_wf = Rzp * v_fl;
v_fr_wf = Rzp * v_fr;

alpha_fl = -atan2(v_fl_wf(2), v_fl_wf(1));
alpha_fr = -atan2(v_fr_wf(2), v_fr_wf(1));
alpha_rl = -atan2(v_rl(2), v_rl(1));
alpha_rr = -atan2(v_rr(2), v_rr(1));

Fx_data = readtable("./data/tire.xlsx", "Sheet", "Fx");
Fx_data = reshape(Fx_data.Fx_N, 10, []).';

Fy_data = readtable("./data/tire.xlsx", "Sheet", "Fy");
Fy_data = abs(reshape(Fy_data.Fy_N, 10, []).');

Fz_range = linspace(100, 1000, 10);
kappa_range = linspace(0, 0.9, 10);
alpha_range = linspace(0, pi/2, 10);

Fx_fcn = interpolant('Fx_fcn','bspline', {Fz_range, kappa_range}, Fx_data(:));
Fy_fcn = interpolant('Fy_fcn','bspline', {Fz_range, alpha_range}, Fy_data(:));

Fzf = M*9.81*lr/L;
Fzr = M*9.81*lf/L;

% Tire force front left
Fx_fl = Fx_fcn([Fzf/2, u(2)]);
Fy_fl = Fy_fcn([Fzf/2, alpha_fl]);
F_fl_wf = [Fx_fl; Fy_fl; 0];
F_fl = Rz*F_fl_wf;

% Tire force front right
Fx_fr = Fx_fcn([Fzf/2, u(3)]);
Fy_fr = Fy_fcn([Fzf/2, alpha_fr]);
F_fr_wf = [Fx_fr; Fy_fr; 0];
F_fr = Rz*F_fr_wf;

% Tire force rear left
Fx_rl = Fx_fcn([Fzr/2, u(4)]);
Fy_rl = Fy_fcn([Fzr/2, alpha_rl]);
F_rl = [Fx_rl; Fy_rl; 0];

% Tire force rear right
Fx_rr = Fx_fcn([Fzr/2, u(5)]);
Fy_rr = Fy_fcn([Fzr/2, alpha_rr]);
F_rr = [Fx_rr; Fy_rr; 0];

% Total force on chassis
F_total = F_fl + F_fr + F_rl + F_rr;

% Total moment on chassis
Mz_total = cross(r_fl, F_fl) + ...
           cross(r_fr, F_fr) + ...
           cross(r_rl, F_rl) + ...
           cross(r_rr, F_rr);

ode_rhs = [y(4)*cos(y(3)) - y(5)*sin(y(3)); % 1 Global X
           y(4)*sin(y(3)) + y(5)*cos(y(3)); % 2 Global Y
           y(6);                            % 3 Global yaw
           1/M*(F_total(1)+M*y(5)*y(6));    % 4 Vx
           1/M*(F_total(2)-M*y(4)*y(6));    % 5 Vy
           1/Iz*(Mz_total(3));              % 6 Psi_dot
           u(1)];                           % 7 steering

f_ode = Function('f', {y, u}, {ode_rhs}, {'y', 'u'}, {'f'});

k1 = f_ode(y, u);
k2 = f_ode(y+Ts*k1/2.0, u); 
k3 = f_ode(y+Ts*k2/2.0, u);
k4 = f_ode(y+Ts*k3, u);

y_next = y + Ts*(k1 + 2*k2 + 2*k3 + k4)/6;

fd_ode = Function('fd', {y, u}, {y_next}, {'y', 'u'}, {'y_next'});

Y = MX.sym('Y', ns, N+1);
U = MX.sym('U', nu, N);
Y_next = fd_ode(Y(:, 1:end-1), U);

Yr = MX.sym('Yr', ns, N);
Y0 = MX.sym('Y0', ns);

nlp = struct;
nlp.x = [vec(Y); vec(U)];
nlp.f = bilin(BigQ, vec(Y(:, 2:end) - Yr)) + bilin(BigR, vec(U)) + bilin(QN, Y(:, end) - Yr(:, end));
nlp.p = [Y0; vec(Yr)];
nlp.g = [Y(:, 1) - Y0; 
         vec(Y(:, 2:end))-vec(Y_next)];

ymax = [Inf, Inf, Inf, 100, Inf, Inf, deg2rad(35)].';
ymin = [-Inf, -Inf, -Inf, 0.1, -Inf, -Inf, -deg2rad(35)].';

Ymax = repmat(ymax, 1, N+1);
Ymin = repmat(ymin, 1, N+1);

umax = [deg2rad(10), 1, 1, 1, 1].';
umin = -umax;

Umax = repmat(umax, 1, N);
Umin = repmat(umin, 1, N);

ubx = [reshape(Ymax, [], 1); reshape(Umax, [], 1)];
lbx = [reshape(Ymin, [], 1); reshape(Umin, [], 1)];

ubg = zeros((N+1)*ns,1);
lbg = ubg;

ipopt_opts = struct;
ipopt_opts.ipopt.print_level = 5;
ipopt_opts.ipopt.tol = 1e-2;
ipopt_opts.ipopt.acceptable_tol = 1e-2;
ipopt_opts.ipopt.linear_solver = 'ma57';
% ipopt_opts.ipopt.hessian_approximation = 'limited-memory';
% ipopt_opts.ipopt.max_iter = 100;

solver = nlpsol('solver', 'ipopt', nlp, ipopt_opts);

ref_traj = berlin_2018(Ts);
ref_traj = [ref_traj; zeros(ns-8, size(ref_traj, 2))];
ref_traj = ref_traj(:, 1:50);

bound1 = readmatrix('./global_racetrajectory_optimization/bound1.csv');
bound2 = readmatrix('./global_racetrajectory_optimization/bound2.csv');
refline = readmatrix('./global_racetrajectory_optimization/refline.csv');
% 
% plot(refline(:, 1), refline(:, 2), 'Color','blue');
% hold on;
% plot(ref_traj(1, :), ref_traj(2, :), 'Color','green');
% plot(bound1(:, 1), bound1(:, 2), 'Color','magenta');
% plot(bound2(:, 1), bound2(:, 2), 'Color','black');

Y0_num = ref_traj(:, 1);
Y0_num(7:10) = Y0_num(4)/reff;

num_pts = size(ref_traj, 2);

sol = solver('x0', zeros(nu*N+ns*(N+1), 1), ...
      'p', [Y0_num; reshape(ref_traj(:, 1:N), [], 1)], ...
      'ubx', ubx, 'lbx', lbx, 'ubg', ubg, 'lbg', lbg);

sol = full(sol.x);

Yopt = reshape(sol(1:(N+1)*ns), ns, []);
Uopt = reshape(sol((N+1)*ns+1:end), nu, []);

i = 1;
j = 1;

while i+N-1 < num_pts

    state_history(:, j) = Y0_num;

    Yr_num = ref_traj(:, i:i+N-1);

    sol = solver('x0', sol, ...
          'p', [Y0_num; reshape(Yr_num, [], 1)], ...
          'ubx', ubx, 'lbx', lbx, 'ubg', ubg, 'lbg', lbg);

    sol = full(sol.x);

    Yopt = reshape(sol(1:(N+1)*ns), ns, []);
    Uopt = reshape(sol((N+1)*ns+1:end), nu, []);

    control_history(:, j) = [Yopt(11:ns, 1); Uopt(1:nu, 1)];

    Y0_num = full(fd_ode(Y0_num, Uopt(1:nu)));

    i = find_closest_index(Y0_num(1), Y0_num(2), ref_traj(1:2, :).')

    j = j + 1;

end

acceleration_history = full(f_ode(state_history, control_history(3:4, :)));

numpts = size(control_history, 2);

figure
plot(state_history(1,:), state_history(2,:), 'linewidth', 3)
xlabel("X (m)");
ylabel("Y (m)");
hold on
plot(ref_traj(1, 1:end-N), ref_traj(2, 1:end-N), '--', 'linewidth', 1.5)

figure
plot(linspace(0, Ts*(numpts-1), numpts), state_history(4,:), 'linewidth', 3)
xlabel("time (s)");
ylabel("v_x (m/s)");

figure
plot(linspace(0, Ts*(numpts-1), numpts), control_history(1, :))
xlabel("time (s)")
ylabel("steering (rad)")

figure
plot(linspace(0, Ts*(numpts-1), numpts), control_history(2, :))
xlabel("time (s)")
ylabel("throttle-brake")

figure
plot(linspace(0, Ts*(numpts-1), numpts), control_history(3, :))
xlabel("time (s)")
ylabel("steering rate (rad/sec)")

figure
plot(linspace(0, Ts*(numpts-1), numpts), control_history(4, :))
xlabel("time (s)")
ylabel("throttle-brake rate")

% save("MPC_run.mat", 'Ts', 'state_history', 'control_history', 'acceleration_history')




