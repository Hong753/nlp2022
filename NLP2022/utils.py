import os
import time

def prepare_folders():
    os.makedirs("./runs", exist_ok=True)
    for name in ["checkpoints", "images", "moments"]:
        os.makedirs(os.path.join("./runs", name), exist_ok=True)
        if name in ["moments"]:
            continue
        for data in ["CUB_200_2011"]:
            for model in ["dfgan", "dfgan_attr"]:
                os.makedirs(os.path.join("./runs", name, "{}_{}".format(data, model)), exist_ok=True)

class MetricLogger():
    def __init__(self, metric_list):
        self.metric_list = metric_list
        self.start_time = time.time()
        for metric in metric_list:
            assert isinstance(metric, str)
            setattr(self, metric, AverageMeter(metric))
    
    def reset(self):
        for metric in self.metric_list:
            getattr(self, metric).reset()

    def print_progress(self):
        for metric in self.metric_list:
            print("{}: {:.4f}".format(metric, getattr(self, metric).avg), end="\t")
        time_taken = time.time() - self.start_time
        print("{:.2f} mins".format(time_taken / 60))

class AverageMeter():
    def __init__(self, name):
        self.name = name
        self.reset()
    
    def reset(self):
        self.sum = 0
        self.count = 0
        
    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        
    @property
    def avg(self):
        if self.count == 0:
            return 0
        else:
            return self.sum / self.count