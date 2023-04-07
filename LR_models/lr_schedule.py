import math


class lr_schedule_cosine():
    def __init__(self, T_0, T_mult=1, eta_max=1., eta_min=0., last_epoch=-1, restart=True,
                 warm_up_schedule=None):
        if warm_up_schedule is None:
            warm_up_schedule = []
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.eta_max = eta_max
        self.T_cur = last_epoch
        self.lr_warm_up_schedule = warm_up_schedule
        self.warm_up_step = len(self.lr_warm_up_schedule)
        self.T_0 = T_0-len(self.lr_warm_up_schedule)-1
        self.T_i = T_0-len(self.lr_warm_up_schedule)-1
        self.restart = restart

    def compute_restart(self):
        self.T_cur += 1
        if self.T_cur == self.T_i:
            lr = self.eta_min
            self.T_i *= self.T_mult
            self.T_cur = -1
        elif self.T_cur == 0:
            lr = self.eta_max
        else:
            lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
        return lr

    def compute(self):
        self.T_cur += 1
        if self.T_cur == 0:
            lr = self.eta_max
        else:
            lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * self.T_cur / self.T_i)) / 2
        return lr

    def get_lr(self, epoch):
        if epoch < len(self.lr_warm_up_schedule):
            lr_select = self.lr_warm_up_schedule[epoch]
        else:
            if self.restart:
                lr_select = self.compute_restart()
            else:
                lr_select = self.compute()
        return lr_select
