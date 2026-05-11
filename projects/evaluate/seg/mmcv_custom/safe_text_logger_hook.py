from mmcv.runner import HOOKS
from mmcv.runner.hooks.logger.text import TextLoggerHook


@HOOKS.register_module(force=True)
class SafeTextLoggerHook(TextLoggerHook):
    """TextLoggerHook that tolerates missing 'data_time' or 'time' keys.

    When EvalHook clears log_buffer mid-iteration (after logging but before
    IterTimerHook re-records), the subsequent LoggerHook firing can find
    data_time missing. This subclass makes the format string resilient.
    """

    def _log_info(self, log_dict, runner):
        if log_dict['mode'] == 'train':
            log_dict.setdefault('data_time', 0.0)
            log_dict.setdefault('time', 0.0)
        super()._log_info(log_dict, runner)
