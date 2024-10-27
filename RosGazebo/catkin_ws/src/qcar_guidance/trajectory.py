import math
import numpy as np
import time
from path import Path
from scipy.optimize import Bounds, minimize, minimize_scalar
from velocity import VelocityProfile


class Trajectory:
  """
  Stores the geometry and dynamics of a path, handling optimisation of the
  racing line. Samples are taken every metre.
  """

  def __init__(self, track, vehicle, initial_velocity=0.0):
    self.track = track
    self.ns = math.ceil(track.length * 40)
    self.update(np.full(track.size, 0.5))
    self.vehicle = vehicle
    self.velocity = None
    self.initial_velocity = initial_velocity

  def get_trajectory_data(self):
    positions = self.path.position(self.s)
    velocities = self.velocity.v

    # Calculate times based on the velocity profile
    times = np.zeros_like(velocities)
    for i in range(1, len(times)):
      avg_velocity = (velocities[i - 1] + velocities[i]) / 2
      times[i] = times[i - 1] + (self.s[i] - self.s[i - 1]) / avg_velocity if avg_velocity != 0 else times[i - 1]

    return times, positions[0], positions[1], velocities

  def _calculate_lap_time(self, alphas):
    self.update(alphas)
    self.update_velocity()
    return self.lap_time()

  def minimise_lap_time(self, max_iterations=100, tolerance=1e-8):
    """
    Generate a path that directly minimises lap time with improved optimization.
    """
    t0 = time.time()

    # Initial guess: mixture of centerline and previous best
    x0 = np.full(self.track.size, 0.5)
    if hasattr(self, 'best_alphas'):
      x0 = 0.5 * (x0 + self.best_alphas)

    # Define bounds to keep alphas between 0 and 1
    bounds = Bounds(0.0, 1.0)

    # Use SLSQP method which supports constraints
    res = minimize(
      fun=self._calculate_lap_time,
      x0=x0,
      method='SLSQP',
      bounds=bounds,
      options={'maxiter': max_iterations, 'ftol': tolerance}
    )

    if res.success:
      new_alphas = res.x
    else:
      print("Optimization failed. Using best known solution.")
      new_alphas = res.x  # Use the result anyway as it might be better than initial

    new_lap_time = self._calculate_lap_time(new_alphas)

    if not hasattr(self, 'best_alphas') or new_lap_time < self._calculate_lap_time(self.best_alphas):
      self.best_alphas = new_alphas
      self.update(self.best_alphas)

    end_time = time.time() - t0
    print(f"Lap time optimization completed in {end_time:.2f} seconds")
    print(f"Current lap time: {new_lap_time:.3f} seconds")

    return end_time

  def iterative_lap_time_optimization(self, n_iterations=3):
    """
    Perform multiple iterations of lap time optimization to refine the result.
    """
    total_time = 0
    best_lap_time = float('inf')

    for i in range(n_iterations):
      print(f"Iteration {i + 1}/{n_iterations}")
      iteration_time = self.minimise_lap_time()
      total_time += iteration_time

      current_lap_time = self.lap_time()
      if current_lap_time < best_lap_time:
        best_lap_time = current_lap_time

      print(f"Best lap time so far: {best_lap_time:.3f} seconds")

    print(f"Final optimized lap time: {best_lap_time:.3f} seconds")
    return total_time

  def update(self, alphas):
    """Update control points and the resulting path."""
    self.alphas = alphas
    self.path = Path(self.track.control_points(alphas), self.track.closed)
    # Sample every metre
    self.s = np.linspace(0, self.path.length, self.ns)

  def update_velocity(self):
    """Generate a new velocity profile for the current path."""
    s = self.s
    s_max = self.path.length if self.track.closed else None
    k = self.path.curvature(s)
    self.velocity = VelocityProfile(self.vehicle, s, k, s_max, max_acceleration=1.0,
                                    initial_velocity=self.initial_velocity)

  def lap_time(self):
    """Calculate lap time from the velocity profile."""
    # Ensure self.s and self.velocity.v have the same length
    if len(self.s) != len(self.velocity.v):
      # Use the shorter length to avoid index out of bounds
      min_length = min(len(self.s), len(self.velocity.v))
      s = self.s[:min_length]
      v = self.velocity.v[:min_length]
    else:
      s = self.s
      v = self.velocity.v

    # Calculate time differences
    ds = np.diff(s)
    v_avg = (v[:-1] + v[1:]) / 2  # Average velocity between points
    dt = ds / v_avg

    return np.sum(dt)

  def minimise_curvature(self):
    """Generate a path minimising curvature."""

    def objfun(alphas):
      self.update(alphas)
      return self.path.gamma2(self.s)

    t0 = time.time()
    res = minimize(
      fun=objfun,
      x0=np.full(self.track.size, 0.5),
      method='L-BFGS-B',
      bounds=Bounds(0.0, 1.0)
    )
    self.update(res.x)
    return time.time() - t0