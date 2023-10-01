import casadi as ca

def sigmoid(x):
    gain = 1000
    return 1.0 / (1.0 + ca.exp(-gain*x))

import casadi as ca

def lateral_force(Fz, alpha, B, C, D, E):
    """
    Calculate the lateral (cornering) force using a simplified magic formula tire model with CasADi.

    Parameters:
    - Fz: Vertical load on the tire (N)
    - alpha: Slip angle (radians)
    - B, C, D, E: Tire model coefficients

    Returns:
    - Fy: Lateral force (N)
    """

    # Fy = Fz * D * ca.sin(C * ca.atan(B * alpha - E * (B * alpha)))
    Fy = Fz*D*alpha

    return Fy