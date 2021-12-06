import utils_torch

class AbstractLog(utils_torch.module.AbstractModule):
    def __init__(self, **kw):
        kw.setdefault("DataOnly", True) # Log class do not need param
        super().__init__(**kw)

class AbstractLogAlongEpochBatchTrain(AbstractLog):
    def __init__(self, **kw):
        super().__init__(**kw)
        return

class AbstractLogAlongBatch(AbstractLog):
    def __init__(self, **kw):
        super().__init__(**kw)
        return
