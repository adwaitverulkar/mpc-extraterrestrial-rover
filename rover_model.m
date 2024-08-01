clc, clear, close all;
addpath casadi-3.6.4-windows64-matlab2018b\
import casadi.*

M = 80;
W = M*9.81;
Iz = 80;
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

alpha_fl_fcn = Function('alpha_fl', {y, u}, {alpha_fl});
alpha_fr_fcn = Function('alpha_fl', {y, u}, {alpha_fr});
alpha_rl_fcn = Function('alpha_fl', {y, u}, {alpha_rl});
alpha_rr_fcn = Function('alpha_fl', {y, u}, {alpha_rr});

Fx_data = readtable("./data/tire.xlsx", "Sheet", "Fx");
Fx_data = reshape(Fx_data.Fx_N, 10, []).';
Fx_data = [fliplr(-Fx_data(:, 2:end)), Fx_data];

Fy_data = readtable("./data/tire.xlsx", "Sheet", "Fy");
Fy_data = abs(reshape(Fy_data.Fy_N, 10, []).');
Fy_data = [fliplr(-Fy_data(:, 2:end)), Fy_data];

Fz_range = linspace(100, 1000, 10);
kappa_range = linspace(-0.9, 0.9, 19);
alpha_range = linspace(-pi/2, pi/2, 19);

Fx_fcn = interpolant('Fx_fcn','bspline', {Fz_range, kappa_range}, Fx_data(:));
Fy_fcn = interpolant('Fy_fcn','bspline', {Fz_range, alpha_range}, Fy_data(:));


Fzf = M*9.81*lr/L;
Fzr = M*9.81*lf/L;

% Tire force front left
Fx_fl = Fx_fcn([Fzf/2, u(2)]);
Fy_fl = Fy_fcn([Fzf/2, alpha_fl]);
F_fl_wf = [Fx_fl; Fy_fl; 0];
F_fl = Rz*F_fl_wf;

F_fl_fcn = Function('Fy_fl', {y, u}, {F_fl});

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

ref_traj = generate_path(Ts);

opti = Opti();

Y = opti.variable(ns, N+1);
U = opti.variable(nu, N);
Y_next = fd_ode(Y(:, 1:end-1), U);

Yr = opti.parameter(ns, N);
Y0 = opti.parameter(ns);

obj = bilin(BigQ, vec(Y(:, 2:end) - Yr)) + bilin(BigR, vec(U)) + bilin(QN, Y(:, end) - Yr(:, end));

alpha_max = pi/2;
alpha_min = -alpha_max;

ymax = [Inf, Inf, Inf, 100, Inf, Inf, deg2rad(35)].';
ymin = [-Inf, -Inf, -Inf, 0.1, -Inf, -Inf, -deg2rad(35)].';

umax = [1, 0.8, 0.8, 0.8, 0.8].';
umin = -umax;

opti.minimize(obj);
opti.subject_to(Y(:, 1) == Y0);
opti.subject_to(vec(Y(:, 2:end)) == vec(Y_next));
opti.subject_to(alpha_min <= alpha_fl_fcn(Y(:, 2:end), U) <= alpha_max);
opti.subject_to(alpha_min <= alpha_fr_fcn(Y(:, 2:end), U) <= alpha_max);
opti.subject_to(alpha_min <= alpha_rl_fcn(Y(:, 2:end), U) <= alpha_max);
opti.subject_to(alpha_min <= alpha_rr_fcn(Y(:, 2:end), U) <= alpha_max);
opti.subject_to(ymin <= Y <= ymax);
opti.subject_to(umin <= U <= umax);

opti.solver('ipopt');

Y0_num = ref_traj(:, 30);

num_pts = size(ref_traj, 2);

opti.set_value(Y0, Y0_num+0.1*randn(7,1));
opti.set_value(Yr, ref_traj(:, 31:N+30));

opti.solve()


