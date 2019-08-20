# -*- encoding: utf-8 -*-
'''
 Created on 2019/2/21.
 @author: eddielin
'''
from .server import Server
from .controller import router


def initialize(app, args):
    server = Server(args)
    router(app, server)
