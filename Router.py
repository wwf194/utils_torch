import re

from utils_torch.attrs import *
from utils_torch.python import *

class Router:
    def __init__(self, param=None, data=None):
        if param is not None:
            self.param = param
    def InitFromParam(self, param):
        utils_torch.parse.ParseRoutingStr(param.Routings)
    def forward(self, **kw):
        param = self.param
        States = utils_torch.EmptyPyObj()
        for name in param.In:
            setattr(States, name, kw[name])
        for Routing in param.Routings:
            utils_torch.parse.ParseRouterDynamic(Routing, States)
            for TimeIndex in range(Routing.RepeatTime):
                InputDict = utils_torch.parse.FilterFromPyObj(States, Routing.In)
                OutObj = Routing.Module(InputDict)
                utils_torch.parse.Register2PyObj(OutObj, States, Routing.Out)
        OutDict = {}
        for name in param.Out:
            OutDict[name] = getattr(States, name)
        return OutDict

def BuildRouter(param):
    return utils_torch.Router.Router(param)
