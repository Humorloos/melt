from ray.tune import CLIReporter


class FlushingReporter(CLIReporter):
    def report(self, trials, done, *sys_info):
        print(self._progress_str(trials, done, *sys_info), flush=True)
