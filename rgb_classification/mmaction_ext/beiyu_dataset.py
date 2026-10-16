# Copyright (c) Beiyu badminton dataset benchmark.
from __future__ import annotations

import os.path as osp
from typing import Callable, List, Optional, Union

from mmengine.fileio import exists, list_from_file

from mmaction.registry import DATASETS
from mmaction.datasets.video_dataset import VideoDataset


@DATASETS.register_module()
class BeiyuVideoDataset(VideoDataset):
    """VideoDataset with per-instance frame ranges.

    Annotation format (space-separated):
        relative_video_path start_frame end_frame label_id

    Example:
        2：.../backhand_drive.mp4 598 635 9
    """

    def load_data_list(self) -> List[dict]:
        exists(self.ann_file)
        data_list = []
        fin = list_from_file(self.ann_file)
        for line in fin:
            parts = line.strip().split(self.delimiter)
            if len(parts) < 4:
                raise ValueError(f"Invalid Beiyu annotation line: {line!r}")
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
