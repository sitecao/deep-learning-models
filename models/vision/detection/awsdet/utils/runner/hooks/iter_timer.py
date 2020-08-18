# Copyright (c) Open-MMLab. All rights reserved.
import time

from .hook import Hook
from ..dist_utils import get_dist_info


class IterTimerHook(Hook):

    def before_epoch(self, runner):
        self.t = time.time()
        self.iter_count = 0
        self.rank = get_dist_info()[0]
        self.timer = []

    def before_iter(self, runner):
        runner.log_buffer.update({'data_time': time.time() - self.t})

    def after_iter(self, runner):
        time_taken = time.time() - self.t
        self.timer.append(time_taken)
        self.iter_count += 1
        runner.log_buffer.update({'time': time_taken})
        self.t = time.time()

    def after_epoch(self, runner):
        if self.rank == 0:
            print("Average iteration time till now is {} ".format(sum(self.timer) / len(self.timer)))

    def every_n_iters(self, runner, n):
        if super(IterTimerHook, self).every_n_iters(runner, n):
            if self.rank == 0:
                print("Average iteration time till iteration {} is {} ".format(self.iter_count, (sum(self.timer) / len(self.timer))))


