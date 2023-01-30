"""
Helper class for nested progress bars during pre-tokenization
"""
import re


class DescStr:
    def __init__(self):
        self._desc = ''

    def write(self, instr):
        self._desc += re.sub('\n|\x1b.*|\r', '', instr)

    def read(self):
        ret = self._desc
        self._desc = ''
        return ret

    def flush(self):
        pass
