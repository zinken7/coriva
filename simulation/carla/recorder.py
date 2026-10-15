import carla
import math
import os
from queue import Queue, Empty


class TrajectoryRecorder:
    """Manages RGB camera + ground truth JSON recording.

    Lifecycle: __init__ → start() → [wait_for_frame() + record_step()] × N → get_trajectory()
    Camera is automatically tracked by World.cleanup().
    """

    def __init__(self, world, attach_to, output_dir,
                 resolution=(1024, 768), fov=100, cam_offset=(1.5, 0.0, 1.7),
                 cam_pitch=0.0):
        """
        Args:
            world: World instance.
            attach_to: CARLA actor to attach camera to (usually Ego.carla_actor).
            output_dir: Directory to save frame PNGs.
            resolution: (width, height) image resolution.
            fov: Field of view (degrees).
            cam_offset: (x, y, z) camera offset from actor.
            cam_pitch: Camera pitch angle (degrees).
        """
        self._world = world
        self._attach_to = attach_to
        self._output_dir = output_dir
        self._resolution = resolution
        self._fov = fov
        self._cam_offset = cam_offset
        self._cam_pitch = cam_pitch

        self._camera = None
        self._image_queue = Queue()
        self._trajectory_data = []
        self._frame_count = 0
        self._last_camera_frame = None

        os.makedirs(self._output_dir, exist_ok=True)

    def start(self):
        """Spawn camera and start listening for frames."""
        cam_bp = self._world.get_blueprint('sensor.camera.rgb')
        cam_bp.set_attribute('image_size_x', str(self._resolution[0]))
        cam_bp.set_attribute('image_size_y', str(self._resolution[1]))
        cam_bp.set_attribute('fov', str(self._fov))

        cam_transform = carla.Transform(
            carla.Location(
                x=self._cam_offset[0],
                y=self._cam_offset[1],
                z=self._cam_offset[2]
            ),
            carla.Rotation(pitch=self._cam_pitch)
        )

        self._camera = self._world.spawn_attached(
            cam_bp, cam_transform, self._attach_to
        )

        output_dir = self._output_dir
        queue = self._image_queue
        self._camera.listen(lambda image: self._on_image(image, output_dir, queue))

    def _on_image(self, image, output_dir, queue):
        """Camera callback — save frame and notify via queue."""
        image.save_to_disk(f"{output_dir}/{image.frame:06d}.png")
        queue.put(image.frame)

    def wait_for_frame(self, timeout=2.0):
        """Blocking wait — wait for a new frame from the camera.

        Returns:
            int: Frame number, or None if timeout.
        """
        try:
            frame = self._image_queue.get(timeout=timeout)
            self._frame_count += 1
            self._last_camera_frame = int(frame)
            return frame
        except Empty:
            self._last_camera_frame = None
            return None

    def record_step(self, time_sec, branch, **actors):
        """Record 1 ground truth step.

        Args:
            time_sec: Current time (s).
            branch: Scenario branch name.
            **actors: Actor name → instance with get_ground_truth().
                      Example: record_step(0.5, "AEB", ego=ego, npc=npc)
        """
        entry = {
            "time_sec": round(time_sec, 2),
            "branch": branch,
            # Binds telemetry to the exact camera image consumed for this
            # simulator tick. Protocol preflight rejects missing history
            # images before any counterfactual rollout is generated.
            "camera_frame": self._last_camera_frame,
        }
        for name, actor in actors.items():
            state = actor.get_ground_truth()
            bbox = self._project_actor_bbox(actor)
            if bbox is not None:
                state["bbox_2d"] = bbox
            entry[name] = state

        self._trajectory_data.append(entry)

    def _project_actor_bbox(self, actor):
        """Project the CARLA actor's real 3D bounding box into this RGB frame.

        The resulting clipped pixel rectangle is the single source used by
        the observed-root visibility gate and the released stable-ID overlay.
        """
        if self._camera is None:
            return None
        carla_actor = getattr(actor, "carla_actor", None)
        bbox = getattr(carla_actor, "bounding_box", None)
        if carla_actor is None or bbox is None:
            return None
        try:
            vertices = bbox.get_world_vertices(carla_actor.get_transform())
            inverse = self._camera.get_transform().get_inverse_matrix()
        except Exception:
            return None

        width, height = self._resolution
        focal = width / (2.0 * math.tan(math.radians(float(self._fov)) / 2.0))
        points = []
        for vertex in vertices:
            point = (float(vertex.x), float(vertex.y), float(vertex.z), 1.0)
            camera = [sum(float(inverse[row][col]) * point[col] for col in range(4))
                      for row in range(4)]
            # CARLA/Unreal camera coordinates -> conventional camera axes.
            depth = camera[0]
            if depth <= 0.1:
                continue
            u = width / 2.0 + focal * camera[1] / depth
            v = height / 2.0 - focal * camera[2] / depth
            points.append((u, v))
        if len(points) < 4:
            return None
        x1 = max(0.0, min(p[0] for p in points))
        y1 = max(0.0, min(p[1] for p in points))
        x2 = min(float(width - 1), max(p[0] for p in points))
        y2 = min(float(height - 1), max(p[1] for p in points))
        if x2 <= x1 or y2 <= y1:
            return None
        return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]

    def record_collision(self, time_sec, branch, extra_frames=5, partners=None):
        """Mark the last recorded step as the collision and capture extra frames.

        Args:
            time_sec: Collision time.
            branch: Branch name.
            extra_frames: Number of extra frames to allow camera to finish capturing.
            partners: Actor names whose footprint overlapped the ego at this
                      tick, in detector order. The reference builder uses
                      them as the impacted actors; never leave them out.
        """
        if self._trajectory_data:
            self._trajectory_data[-1]["collision"] = True
            self._trajectory_data[-1]["collision_time"] = round(time_sec, 2)
            self._trajectory_data[-1]["collision_partners"] = list(partners or [])

        # Run N more ticks for camera capture
        for _ in range(extra_frames):
            self._world.tick()
            self.wait_for_frame()

    def get_trajectory(self):
        """Return the list of recorded ground truth data."""
        return self._trajectory_data

    def get_frame_count(self):
        """Number of frames successfully captured by camera."""
        return self._frame_count

    def stop(self):
        """Stop camera sensor."""
        if self._camera and self._camera.is_alive:
            self._camera.stop()
