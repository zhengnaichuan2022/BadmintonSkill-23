# Restored from bytecode backup + 5-node patch for BadmintonSkill-23.
from __future__ import annotations

import marshal
from pathlib import Path

import numpy as np

_PYC = Path(__file__).resolve().parents[2] / "_pyc_backup" / "tsf" / "utils.cpython-38.pyc"
exec(marshal.loads(_PYC.read_bytes()[16:]), globals())


def extract_imu_tensor_func_multinode(in_x, version):
    """BadmintonSkill-23 IMU: single-node 6-ch or multi-node 6×N (5 nodes → 30 channels).

    Channel layout per node: [acc.x/y/z, gyro.x/y/z].
    """
    n_ch = int(in_x.shape[-1])
    if n_ch % 6 != 0:
        raise ValueError(
            f"BadmintonSkill-23 IMU expects channels multiple of 6 (acc+gyro per node), got {n_ch}"
        )

    n_sensors = n_ch // 6
    x_imu = []
    tags = []
    sensor_location = 1
    for i in range(n_sensors):
        x_imu.append(in_x[:, :, i * 6 : (i + 1) * 6])
        l = []
        for flag in (Tag.ACC, Tag.GYRO):
            l.extend([[sensor_location, flag, axis] for axis in Tag.XYZ])
        tags.append(np.array(l))
        sensor_location += 1

    return None, None, [x_imu], [tags]
