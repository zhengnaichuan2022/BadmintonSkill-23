# Copyright (c) BadmintonSkill-23 benchmark.
from __future__ import annotations

import os.path as osp
from typing import List

from mmengine.fileio import exists, list_from_file

from mmaction.registry import DATASETS
from mmaction.datasets.video_dataset import VideoDataset


@DATASETS.register_module()
class BadmintonSkill23VideoDataset(VideoDataset):
    """VideoDataset with per-instance frame ranges for BadmintonSkill-23.

    Annotation format (space-separated):
        relative_video_path start_frame end_frame label_id

    Example:
        sessions/S001_1028/rgb/A01_forehand_net.mp4 598 635 0
    """

    def load_data_list(self) -> List[dict]:
        exists(self.ann_file)
        data_list = []
        fin = list_from_file(self.ann_file)
        for line in fin:
            parts = line.strip().split(self.delimiter)
            if len(parts) < 4:
                raise ValueError(f"Invalid BadmintonSkill-23 annotation line: {line!r}")
            filename, start_frame, end_frame, label = parts[:4]
            label = int(label)
            start_frame = int(start_frame)
            end_frame = int(end_frame)
            if self.data_prefix["video"] is not None:
                filename = osp.join(self.data_prefix["video"], filename)
            data_list.append(
                dict(
                    filename=filename,
                    label=label,
                    start_frame=start_frame,
                    end_frame=end_frame,
                )
            )
        return data_list
