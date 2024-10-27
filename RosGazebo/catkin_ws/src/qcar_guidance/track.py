import json
import numpy as np
from scipy.interpolate import splprep, splev

from path import Path
from utils import define_corners, idx_modulo, is_closed


class Track:
    """Represents a track with boundaries defined by a series of cones."""

    def __init__(self, json_path=None, left=None, right=None, car_width=0, start_position=None):
        """Create track from cone coordinates."""
        self.car_width = car_width
        if json_path is None:
            # Ensure left and right are numpy arrays
            self.left = np.array(left)
            self.right = np.array(right)
        else:
            self.read_cones(json_path)
        self.shrink_boundaries()
        self.closed = is_closed(self.left, self.right)
        self.size = self.left.shape[1] - int(self.closed)
        self.diffs = self.right - self.left
        self.mid = Path(self.control_points(np.full(self.size, 0.5)), self.closed)
        if start_position is not None:
            self.shift_controls_to_start_at_point(start_position[0], start_position[1])
        self.length = self.mid.dists[-1]
        self.smooth_midline()

    def shift_controls_to_start_at_point(self, x0, y0):
        """Shift the track so that the midline starts at the point closest to (x0, y0)."""
        # Compute midline control points
        mid_controls = self.control_points(np.full(self.size, 0.5))
        # Compute distances from midline control points to (x0, y0)
        distances = np.sqrt((mid_controls[0] - x0) ** 2 + (mid_controls[1] - y0) ** 2)
        # Find the index with the minimum distance
        min_index = np.argmin(distances)
        # Rotate left, right, and diffs so that min_index is at the start
        self.left = np.roll(self.left, -min_index, axis=1)
        self.right = np.roll(self.right, -min_index, axis=1)
        self.diffs = np.roll(self.diffs, -min_index, axis=1)
        # Recompute midline
        self.mid = Path(self.control_points(np.full(self.size, 0.5)), self.closed)

    def shrink_boundaries(self):
        """Shrink the track boundaries based on the car's width."""
        midline = (self.left + self.right) / 2
        left_vector = self.left - midline
        right_vector = self.right - midline

        # Normalize vectors
        left_norm = np.linalg.norm(left_vector, axis=0)
        right_norm = np.linalg.norm(right_vector, axis=0)
        left_unit = left_vector / left_norm
        right_unit = right_vector / right_norm

        # Shrink boundaries
        self.left = midline + left_unit * (left_norm - self.car_width / 2)
        self.right = midline + right_unit * (right_norm - self.car_width / 2)

    def smooth_midline(self, smoothing_factor=0.1):
        # Extract x and y coordinates of the midline
        x = self.mid.controls[0]
        y = self.mid.controls[1]

        # Perform smoothing using spline interpolation
        tck, u = splprep([x, y], s=smoothing_factor, per=self.closed)

        # Generate new points
        new_u = np.linspace(0, 1, len(x))
        new_points = splev(new_u, tck)

        # Update the midline with smoothed points
        self.mid = Path(np.array(new_points), self.closed)

    def read_cones(self, path):
        """Read cone coordinates from a JSON file."""
        track_data = json.load(open(path))
        self.left = np.array([track_data["left"]["x"], track_data["left"]["y"]])
        self.right = np.array([track_data["right"]["x"], track_data["right"]["y"]])
        print("[ Imported Track ]")

    def avg_curvature(self, s):
        """Return the average of curvatures at the given sample distances."""
        k = self.mid.curvature(s)
        return np.sum(k) / s.size

    def corners(self, s, k_min, proximity, length):
        """Determine location of corners on this track."""
        return define_corners(self.mid, s, k_min, proximity, length)

    def control_points(self, alphas):
        """Translate alpha values to control point coordinates."""
        if self.closed:
            alphas = np.append(alphas, alphas[0])
        i = np.nonzero(alphas != -1)[0]
        return self.left[:, i] + (alphas[i] * self.diffs[:, i])
