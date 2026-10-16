# DEPRECATED — use auto-generated benchmark configs instead:
#   configs/beiyu/benchmark/rgb_skel_4stream_xsub_actionformer.py  (2048-d)
#   configs/beiyu/benchmark/rgb_skel_jb_xsub_actionformer.py       (1280-d)
#   bash run_tad_rgb_skel_benchmark.sh
#
# Old I3D+STGCN paths are obsolete. Regenerate with:
#   python tools/generate_beiyu_tad_benchmark.py

_base_ = [
    "../benchmark/rgb_skel_4stream_xsub_actionformer.py",
]
