from models.models import rover_model_nlbicycle
import numpy as np
import casadi as ca

rover_params = {
                'M': 1000.0,       # Mass of the rover (kg)
                'Iz': 2000.0,      # Moment of inertia (kg*m^2)
                'lf': 2.0,         # Distance from the center of mass to the front axle (m)
                'lr': 2.0,         # Distance from the center of mass to the rear axle (m)
                'h': 0.5,          # Height of the center of mass (m)
                'eps': 0.1,        # Small constant (m)
                'acc_sl': 5000.0,  # Acceleration slip coefficient
                'brake_sl': 5000.0, # Braking slip coefficient
                'Bf': 10.0,        # Tire model coefficient
                'Cf': 1.5,         # Tire model coefficient
                'Df': 10000.0,     # Tire model coefficient
                'Ef': 0.3,         # Tire model coefficient
                'Br': 10.0,        # Tire model coefficient
                'Cr': 1.5,         # Tire model coefficient
                'Dr': 10000.0,     # Tire model coefficient
                'Er': 0.3,         # Tire model coefficient
                'Ts': 0.001         # Time step for the model (s)
                }
x0 = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
u0 = np.array([0.0, 0.0])

# Create an instance of the rover model with the parameter dictionary
rover_model = rover_model_nlbicycle(rover_params)

print(rover_model.model_ms(x0, u0))
print(rover_model.model_ss(x0, u0))