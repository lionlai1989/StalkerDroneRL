"""Use gz-transport for Gazebo service requests, replacing subprocess-based `gz service` calls
that often fail with: "NodeShared::RecvSrvRequest() error sending response: Host unreachable".
See: https://github.com/gazebosim/gz-transport/issues/564
"""

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
    req.position.x = x
    req.position.y = y
    req.position.z = z
    req.orientation.w = qw
    req.orientation.x = qx
    req.orientation.y = qy
    req.orientation.z = qz
    if not _request(f"/world/{world}/set_pose", req, timeout_ms):
        raise RuntimeError("set_pose failed")
