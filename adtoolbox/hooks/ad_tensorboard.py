from datetime import datetime

from mmcv import TORCH_VERSION, digit_version
from mmcv.runner import TensorboardLoggerHook, master_only, HOOKS
import os.path as osp


@HOOKS.register_module()
class ADTensorboardLoggerHook(TensorboardLoggerHook):

    @master_only
    def before_run(self, runner):
        super(TensorboardLoggerHook, self).before_run(runner)
        if (TORCH_VERSION == 'parrots'
                or digit_version(TORCH_VERSION) < digit_version('1.1')):
            try:
                from tensorboardX import SummaryWriter
            except ImportError:
                raise ImportError('Please install tensorboardX to use '
                                  'TensorboardLoggerHook.')
        else:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')

        if self.log_dir is None:
            current_time = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            self.log_dir = osp.join(runner.work_dir, 'tf_logs', current_time)
        self.writer = SummaryWriter(self.log_dir)
