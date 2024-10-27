from math import sqrt
import json
import numpy as np

GRAV = 9.81  # m/s^2

###############################################################################

class Vehicle:
  """Vehicle parameters and behaviour."""

  def __init__(self, vehicle_enum):
    """Load vehicle data from JSON file."""
    # vehicle_data = json.load(open(path))
    config = vehicle_enum.value
    self.name = config["name"]
    self.mass = config["mass"]
    self.cof = config["frictionCoefficient"]
    self.width = config["width"]  # Set the width of the vehicle in meters
    self.engine_profile = [
      config["engineMap"]["v"],
      config["engineMap"]["f"]
    ]
    print("[ Imported {} ]".format(self.name))


  def engine_force(self, velocity, gear=None):
    """Map current velocity to force output by the engine."""
    return np.interp(velocity, self.engine_profile[0], self.engine_profile[1])


  def traction(self, velocity, curvature):
    """Determine remaining traction when negotiating a corner."""
    f = self.cof * self.mass * GRAV
    f_lat = self.mass * velocity**2 * curvature
    if f <= f_lat: return 0
    return sqrt(f**2 - f_lat**2)

