# Copyright (c) Open-MMLab. All rights reserved.
import time

from .hook import Hook
from ..dist_utils import get_dist_info
import math

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
                print("Current step time is {}, Average iteration time till iteration {} is {} ".format(self.timer[-1], self.iter_count, (sum(self.timer) / len(self.timer))))
                print ("p50 step time is {}".format(self.percentile(50)))

    def percentile(self, percentile):
        size = len(self.timer)
        return sorted(self.timer)[int(math.ceil((size * percentile) / 100)) - 1]
