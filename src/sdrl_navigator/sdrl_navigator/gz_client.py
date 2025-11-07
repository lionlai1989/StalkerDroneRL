"""Use gz-transport for Gazebo service requests, replacing subprocess-based `gz service` calls
that often fail with: "NodeShared::RecvSrvRequest() error sending response: Host unreachable".
See: https://github.com/gazebosim/gz-transport/issues/564
"""

import subprocess
import time
from pathlib import Path

try:
    # Preferred (unversioned) imports if available
    import gz.transport as gz_transport
    from gz.msgs.boolean_pb2 import Boolean
    from gz.msgs.pose_pb2 import Pose
    from gz.msgs.world_control_pb2 import WorldControl
except ModuleNotFoundError:
    # Harmonic Debian packages expose versioned subpackages
    import gz.transport13 as gz_transport
    from gz.msgs10.boolean_pb2 import Boolean
    from gz.msgs10.pose_pb2 import Pose
    from gz.msgs10.world_control_pb2 import WorldControl

_NODE = gz_transport.Node()


def _request(service: str, req, timeout_ms: int) -> bool:
    """Send a service request using whichever binding signature is available.

    Tries (newer gz-transport13):
      request(service, request_msg, request_type, response_type, timeout_ms)
    """
    # Preferred signature (gz.transport13): pass message types
    res = _NODE.request(service, req, req.__class__, Boolean, int(timeout_ms))
    # Some bindings return (ok, resp), others just resp (Boolean)
    if isinstance(res, tuple):
        ok, resp = res
        return bool(ok and getattr(resp, "data", False))
    return bool(getattr(res, "data", False))


def world_control(
    world: str, *, pause: bool | None = None, step_multi: int | None = None, timeout_ms: int = 10000
) -> None:
    req = WorldControl()
    if pause is not None:
        req.pause = bool(pause)
    if step_multi is not None:
        req.step = True
        req.multi_step = int(step_multi)
    if not _request(f"/world/{world}/control", req, timeout_ms):
        raise RuntimeError("world_control failed")


def set_pose(
    world: str,
    model: str,
    *,
    x: float,
    y: float,
    z: float,
    qw: float,
    qx: float,
    qy: float,
    qz: float,
    timeout_ms: int = 10000,
) -> None:
    req = Pose()
    req.name = model
    req.position.x = float(x)
    req.position.y = float(y)
    req.position.z = float(z)
    req.orientation.w = float(qw)
    req.orientation.x = float(qx)
    req.orientation.y = float(qy)
    req.orientation.z = float(qz)
    if not _request(f"/world/{world}/set_pose", req, timeout_ms):
        raise RuntimeError("set_pose failed")


def respawn_drone():
    """Remove lion_quadcopter model from the world and respawn it.
    Do not use this. Repositioning the drone to initial pose is better."""
    remove_cmd = [
        "gz",
        "service",
        "-s",
        "/world/ground_plane_world/remove",
        "--reqtype",
        "gz.msgs.Entity",
        "--reptype",
        "gz.msgs.Boolean",
        "--timeout",
        "3000",
        "--req",
        'name: "lion_quadcopter" type: MODEL',
    ]
    rc1 = subprocess.run(remove_cmd, check=False, capture_output=True, text=True)
    if rc1.returncode != 0:
        raise RuntimeError(f"Remove drone failed (code {rc1.returncode}): {rc1.stderr}")

    time.sleep(0.2)  # gives Gazebo a moment to fully tear down the model/plugins.

    lion_sdf_path = Path("models") / "lion_quadcopter.sdf"
    lion_sdf_pose = (
        "pose: { position: { x: 0, y: 0, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } }"
    )
    spawn_cmd = [
        "gz",
        "service",
        "-s",
        "/world/ground_plane_world/create",
        "--reqtype",
        "gz.msgs.EntityFactory",
        "--reptype",
        "gz.msgs.Boolean",
        "--timeout",
        "3000",
        "--req",
        f'sdf_filename: "{lion_sdf_path}" name: "lion_quadcopter" {lion_sdf_pose}',
    ]
    rc2 = subprocess.run(spawn_cmd, check=False, capture_output=True, text=True)
    if rc2.returncode != 0:
        raise RuntimeError(f"Spawn drone failed (code {rc2.returncode}): {rc2.stderr}")
