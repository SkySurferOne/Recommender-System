import inspect

from src.main.utils.LogTime import LogTime


class Timed:

    def timed(func):
        caller = inspect.stack()[1][3]

        def wrapped(self, *args):
            if hasattr(self, 'verbose') and self.verbose:
                time = LogTime('{}.{}'.format(caller, func.__name__))
                res = func(self, *args)
                time.finish()
                return res
            else:
                return func(self, *args)

        return wrapped

    timed = staticmethod(timed)