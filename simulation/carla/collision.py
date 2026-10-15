class CollisionDetector:
    """Axis-aligned-in-ego-frame footprint collision detector.

    Every CORIVA actor carries a body length and width.  Using those extents
    makes parked vehicles, large vehicles, bicycles, pedestrians and static
    obstacles participate consistently instead of sharing one car-sized
    threshold.
    """

    def __init__(self, actor_a, actor_b,
                 length_threshold=4.5, width_threshold=2.0):
        """
        Args:
            actor_a: First actor (Ego/NPC with road_fwd, road_left).
            actor_b: Second actor (Ego/NPC or Pedestrian).
            length_threshold: Longitudinal distance threshold (m). For pedestrians,
                              used as collision radius.
            width_threshold: Lateral distance threshold (m).
        """
        self.actor_a = actor_a
        self.actor_b = actor_b
        a_length = float(getattr(actor_a, "bounding_length", length_threshold))
        b_length = float(getattr(actor_b, "bounding_length", length_threshold))
        a_width = float(getattr(actor_a, "bounding_width", width_threshold))
        b_width = float(getattr(actor_b, "bounding_width", width_threshold))
        self.length_threshold = (a_length + b_length) / 2.0
        self.width_threshold = (a_width + b_width) / 2.0

    def get_distances(self):
        """Calculate distances between 2 actors.

        Returns:
            dict: {"forward": float, "lateral": float} for vehicle-vehicle,
                  {"distance": float} for vehicle-pedestrian.
        """
        dx = self.actor_a.x - self.actor_b.x
        dy = self.actor_a.y - self.actor_b.y

        fwd = self.actor_a.road_fwd
        left = self.actor_a.road_left
        dist_fwd = abs(dx * fwd.x + dy * fwd.y)
        dist_lat = abs(dx * left.x + dy * left.y)
        return {"forward": round(dist_fwd, 3), "lateral": round(dist_lat, 3)}

    def check(self):
        """Check for collision.

        Returns:
            bool: True if the 2 actors overlap.
        """
        distances = self.get_distances()

        return (distances["forward"] < self.length_threshold and
                distances["lateral"] < self.width_threshold)
