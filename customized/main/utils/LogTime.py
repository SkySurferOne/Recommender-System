import time


class LogTime:

    def __init__(self, label=''):
        self.start_time = time.time()
        self.label = label

    def finish(self):
        total_time = self.get_total_time()
        if self.label == '':
            print('total %.2f seconds have spent\n' % total_time)
        else:
            print('[%s] total %.2f seconds have spent\n' % (self.label, total_time))

    def get_total_time(self):
        return time.time() - self.start_time
