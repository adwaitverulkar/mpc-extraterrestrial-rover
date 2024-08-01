function Yr = generate_path(Ts)

    ref_x = (0:100).';
    ref_y = zeros(size(ref_x));
    ref_z = zeros(size(ref_x));

    ref_vx = 2*ones(size(ref_x));
    
    ref_traj = waypointTrajectory([ref_x ref_y ref_z], GroundSpeed=ref_vx);
    ref_traj.SampleRate = 1/Ts;
    
    num_pts = round(ref_traj.TimeOfArrival(end)*ref_traj.SampleRate);
    ref_traj.SamplesPerFrame = num_pts;


    [position, orientation, velocity, ~, angularVelocity] = ref_traj();

    Yr(1:2, :) = position(1:end-1, 1:2).';
    
    eulerAngles = quat2eul(orientation, 'ZYX');
    Yr(3, :) = unwrap(eulerAngles(1:end-1, 1).');

    Yr(4, :) = vecnorm(velocity(1:end-1, :).');
    Yr(5, :) = zeros(1, num_pts-1);
    Yr(6, :) = angularVelocity(1:end-1, 3).';

    Yr(7, :) = zeros(1, num_pts-1);

end