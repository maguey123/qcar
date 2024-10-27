import numpy as np
from math import sqrt

GRAV = 9.81  # ms^-2


class VelocityProfile:
    """
    Stores and generates a velocity profile for a given path and vehicle.
    """

    def __init__(self, vehicle, s, k, s_max=None, max_acceleration=None, initial_velocity=0.0):
        """
        Generate a velocity profile for the given vehicle and path parameters.
        :s: and :k: should NOT include the overlapping element for closed paths.
        The length of a closed path should be supplied in :s_max:
        :max_acceleration: is the maximum acceleration of the vehicle in m/s^2
        :initial_velocity: is the starting velocity of the vehicle in m/s
        """
        self.vehicle = vehicle
        self.s = s
        self.s_max = s_max
        self.k = k
        self.max_acceleration = max_acceleration if max_acceleration is not None else float('inf')
        self.initial_velocity = initial_velocity
        self.max_velocity = self.get_max_velocity()
        self.v_local = self.limit_local_velocities()
        self.v_acclim = self.limit_acceleration()
        self.v_declim = self.limit_deceleration()
        self.v = np.minimum(self.v_acclim, self.v_declim)
        self.enforce_max_velocity()
        self.enforce_initial_velocity()

    def get_max_velocity(self):
        """Determine the maximum velocity from the engine profile."""
        return self.vehicle.engine_profile[0][-1]

    def limit_local_velocities(self):
        v_local = np.sqrt(self.vehicle.cof * GRAV / np.abs(self.k))
        return np.minimum(v_local, self.max_velocity)

    def limit_acceleration(self):
        v = np.zeros_like(self.s)
        v[0] = self.initial_velocity

        for i in range(1, len(self.s)):
            ds = self.s[i] - self.s[i - 1]
            traction = self.vehicle.traction(v[i - 1], self.k[i - 1])
            force = min(self.vehicle.engine_force(v[i - 1]), traction)
            accel = min(force / self.vehicle.mass, self.max_acceleration)

            # Calculate the maximum velocity achievable with the given acceleration
            v_max_accel = sqrt(v[i - 1] ** 2 + 2 * accel * ds)

            # Limit the velocity based on local constraints and acceleration
            v[i] = min(v_max_accel, self.v_local[i], self.max_velocity)

        return v

    def limit_deceleration(self):
        v = self.v_local.copy()

        for i in range(len(self.s) - 2, -1, -1):
            ds = self.s[i + 1] - self.s[i]
            traction = self.vehicle.traction(v[i + 1], self.k[i + 1])
            decel = min(traction / self.vehicle.mass, self.max_acceleration)

            v_max_decel = sqrt(v[i + 1] ** 2 + 2 * decel * ds)
            v[i] = min(v[i], v_max_decel, self.max_velocity)

        return v

    def enforce_max_velocity(self):
        """Ensure that no velocity exceeds the maximum velocity."""
        self.v = np.minimum(self.v, self.max_velocity)

    def enforce_initial_velocity(self):
        """Ensure that the initial velocity is set correctly."""
        self.v[0] = self.initial_velocity

    def get_velocity_at(self, s):
        """
        Get the velocity at a specific position along the path.
        """
        return np.interp(s, self.s, self.v)