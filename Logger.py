import sys
class Logger():
    def __init__(self):
        self.output_dests = {}
        self.stdout=False
    def add_flow(self, flow, name="undefined"):
        self.output_dests[name]=flow
    def add_stdout(self):
        self.stdout=True
    def write(self, message, end="\n"):
        for output_dest in self.output_dests.values():
            output_dest.write(message + end)
        if self.stdout:
            print(message, end=end)
    def clear(self):
        self.output_dests = {}
        self.stdout=False