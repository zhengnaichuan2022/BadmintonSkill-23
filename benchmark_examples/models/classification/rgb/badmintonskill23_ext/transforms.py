# Copyright (c) BadmintonSkill-23 benchmark.
from __future__ import annotations

from typing import Dict

from mmcv.transforms import BaseTransform

from mmaction.registry import TRANSFORMS
from mmaction.datasets.transforms.loading import DecordInit


@TRANSFORMS.register_module()
class BadmintonSkill23DecordInit(DecordInit):
    """Decord video init; read from disk path directly."""

    def _get_video_reader(self, filename: str) -> object:
        import decord
        from decord import cpu

        return decord.VideoReader(
            filename,
            ctx=cpu(0),
            num_threads=self.num_threads,
        )


@TRANSFORMS.register_module()
class BadmintonSkill23ApplyFrameRange(BaseTransform):
    """Restrict sampling to [start_frame, end_frame] from annotation."""

    def transform(self, results: Dict) -> Dict:
        total = int(results["total_frames"])
        if total <= 0:
            raise ValueError(f"Empty video: {results.get('filename')}")

        start = int(results.get("start_frame", 0))
        end = int(results.get("end_frame", total - 1))
        start = max(0, min(start, total - 1))
        end = max(0, min(end, total - 1))
        if end < start:
            start, end = 0, total - 1

        results["frame_offset"] = start
        results["total_frames"] = end - start + 1
        return results


@TRANSFORMS.register_module()
class BadmintonSkill23OffsetFrameInds(BaseTransform):
    """Shift sampled indices back to absolute positions in source video."""

    def transform(self, results: Dict) -> Dict:
        offset = int(results.get("frame_offset", 0))
        if offset:
            results["frame_inds"] = results["frame_inds"] + offset
        return results
