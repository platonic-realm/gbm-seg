# Python Imports
import os

# Library Imports
import torch
from torch.profiler import ProfilerActivity

# Local Imports


class Profiler():

    def __init__(self,
                 _enabled: bool,
                 _save_path: str,
                 _scheduler_wait: int,
                 _scheduler_warmup: int,
                 _scheduler_active: int,
                 _scheduler_repeat: int,
                 _profile_memory: bool,
                 _record_shapes: bool,
                 _with_flops: bool,
                 _with_stack: bool,
                 _save_tensorboard: bool,
                 _save_text: bool,
                 _save_std: bool):

        self.enabled = _enabled

        tb_trace_handler = \
            torch.profiler.tensorboard_trace_handler(_save_path)

        trace_file = os.path.join(_save_path, "trace.txt")

        def txt_trace_handler(prof):
            with open(trace_file, 'w', encoding='UTF-8') as file:
                file.write(prof.key_averages().table(
                                sort_by="self_cuda_time_total",
                                row_limit=-1))

        def print_trace_handler(prof):
            print(prof.key_averages().table(
                            sort_by="self_cuda_time_total",
                            row_limit=-1))

        def trace_handler(prof):
            if _save_tensorboard:
                tb_trace_handler(prof)

            if _save_text:
                txt_trace_handler(prof)

            if _save_std:
                print_trace_handler(prof)

        self.prof = torch.profiler.profile(
                activities=[ProfilerActivity.CUDA,
                            ProfilerActivity.CPU],
                schedule=torch.profiler.schedule(wait=_scheduler_wait,
                                                 warmup=_scheduler_warmup,
                                                 active=_scheduler_active,
                                                 repeat=_scheduler_repeat),
                on_trace_ready=trace_handler,
                profile_memory=_profile_memory,
                record_shapes=_record_shapes,
                with_flops=_with_flops,
                with_stack=_with_stack) if _enabled else None

    def start(self):
        if self.enabled:
            self.prof.start()

    def stop(self):
        if self.enabled:
            self.prof.stop()

    def step(self):
        if self.enabled:
            self.prof.step()
