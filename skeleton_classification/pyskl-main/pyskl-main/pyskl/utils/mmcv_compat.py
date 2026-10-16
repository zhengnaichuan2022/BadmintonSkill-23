# Copyright (c) OpenMMLab. All rights reserved.
"""Compatibility patches for mmcv 1.3.x with PyTorch >= 1.11 and yapf >= 0.40."""

import torch
from mmcv import print_log
from mmcv.utils import TORCH_VERSION, digit_version
from torch.nn.parallel.distributed import _find_tensors


def apply_mmcv_compat():
    _patch_yapf_format_code()
    _patch_mmdistributed_data_parallel()


def _patch_yapf_format_code():
    """yapf>=0.40 removed the verify kwarg that mmcv 1.x passes to FormatCode."""
    try:
        from yapf.yapflib import yapf_api
    except ImportError:
        return

    if getattr(yapf_api, '_pyskl_compat', False):
        return

    orig_format_code = yapf_api.FormatCode

    def format_code_compat(*args, **kwargs):
        kwargs.pop('verify', None)
        return orig_format_code(*args, **kwargs)

    yapf_api.FormatCode = format_code_compat
    yapf_api._pyskl_compat = True

    try:
        import mmcv.utils.config as mmcv_config
        mmcv_config.FormatCode = format_code_compat
    except ImportError:
        pass


def _patch_mmdistributed_data_parallel():
    """mmcv 1.3.x calls DDP._sync_params(), removed in PyTorch >= 1.11."""
    from mmcv.parallel import distributed as mmcv_dist

    MMDDP = mmcv_dist.MMDistributedDataParallel
    if getattr(MMDDP, '_pyskl_pytorch2_compat', False):
        return

    def sync_before_forward(self):
        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.11.0a0')):
            if self._check_sync_bufs_pre_fwd():
                self._sync_buffers()
        elif getattr(self, 'require_forward_param_sync', True):
            self._sync_params()

    def sync_after_forward(self):
        if ('parrots' not in TORCH_VERSION
                and digit_version(TORCH_VERSION) >= digit_version('1.11.0a0')):
            if self._check_sync_bufs_post_fwd():
                self._sync_buffers()

    def train_step(self, *inputs, **kwargs):
        if (TORCH_VERSION >= '1.7' and 'parrots'
                not in TORCH_VERSION) and self.reducer._rebuild_buckets():
            print_log(
                'Reducer buckets have been rebuilt in this iteration.',
                logger='mmcv')

        sync_before_forward(self)
        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                output = self.module.train_step(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(
                    self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module.train_step(*inputs, **kwargs)

        sync_after_forward(self)

        if torch.is_grad_enabled() and getattr(
                self, 'require_backward_grad_sync', True):
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            if TORCH_VERSION > '1.2':
                self.require_forward_param_sync = False
        return output

    def val_step(self, *inputs, **kwargs):
        if (TORCH_VERSION >= '1.7' and 'parrots'
                not in TORCH_VERSION) and self.reducer._rebuild_buckets():
            print_log(
                'Reducer buckets have been rebuilt in this iteration.',
                logger='mmcv')

        sync_before_forward(self)
        if self.device_ids:
            inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                output = self.module.val_step(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(
                    self._module_copies[:len(inputs)], inputs, kwargs)
                output = self.gather(outputs, self.output_device)
        else:
            output = self.module.val_step(*inputs, **kwargs)

        sync_after_forward(self)

        if torch.is_grad_enabled() and getattr(
                self, 'require_backward_grad_sync', True):
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            if TORCH_VERSION > '1.2':
                self.require_forward_param_sync = False
        return output

    MMDDP.train_step = train_step
    MMDDP.val_step = val_step
    MMDDP._pyskl_pytorch2_compat = True


apply_mmcv_compat()
