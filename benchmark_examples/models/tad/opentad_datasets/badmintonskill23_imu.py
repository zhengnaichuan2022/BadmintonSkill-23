from copy import deepcopy

import numpy as np

from .base import SlidingWindowDataset, filter_same_annotation
from .builder import DATASETS


@DATASETS.register_module()
class BadmintonSkill23IMUSlidingDataset(SlidingWindowDataset):
    """BadmintonSkill-23 IMU end-to-end TAD with sliding windows on raw sample timeline."""

    def get_gt(self, video_info, thresh=0.0):
        gt_segment = []
        gt_label = []
        for anno in video_info["annotations"]:
            label = anno["label"]
            if label not in self.class_map:
                continue
            gt_start = int(anno["segment"][0] / video_info["duration"] * video_info["frame"])
            gt_end = int(anno["segment"][1] / video_info["duration"] * video_info["frame"])
            if (not self.filter_gt) or (gt_end - gt_start > thresh):
                gt_segment.append([gt_start, gt_end])
                gt_label.append(self.class_map.index(label))

        if len(gt_segment) == 0:
            return None
        return filter_same_annotation(
            dict(
                gt_segments=np.array(gt_segment, dtype=np.float32),
                gt_labels=np.array(gt_label, dtype=np.int32),
            )
        )

    def __getitem__(self, index):
        video_name, video_info, video_anno, window_snippet_centers = self.data_list[index]

        if video_anno != {}:
            video_anno = deepcopy(video_anno)
            video_anno["gt_segments"] = video_anno["gt_segments"] - window_snippet_centers[0] - self.offset_frames
            video_anno["gt_segments"] = video_anno["gt_segments"] / self.snippet_stride

        return self.pipeline(
            dict(
                video_name=video_name,
                data_path=self.data_path,
                imu_csv=video_info.get("imu_csv", ""),
                sensor_scale_factor=video_info.get("sensor_scale_factor", 1.0),
                window_size=self.window_size,
                feature_start_idx=int(window_snippet_centers[0] / self.snippet_stride),
                feature_end_idx=int(window_snippet_centers[-1] / self.snippet_stride),
                sample_stride=self.sample_stride,
                fps=video_info["fps"],
                snippet_stride=self.snippet_stride,
                window_start_frame=window_snippet_centers[0],
                duration=video_info["duration"],
                offset_frames=self.offset_frames,
                **video_anno,
            )
        )
