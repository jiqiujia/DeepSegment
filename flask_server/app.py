# -*- encoding: utf-8 -*-
'''
 Created on 2019/2/21.
 @author: eddielin
'''
import logging
from argparse import ArgumentParser, Namespace

import opts

import yaml

from utils import misc_utils

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from flask import Flask, jsonify
from flask_cors import CORS
from gevent.pywsgi import WSGIServer

from flask_server.initialize import initialize

def start(args,
          url_root="./nlp",
          host="0.0.0.0",
          port=8080):
    def prefix_route(route_function, prefix='', mask='{0}{1}'):
        def newroute(route, *args, **kwargs):
            return route_function(mask.format(prefix, route), *args, **kwargs)

        return newroute

    app = Flask(__name__)
    app.route = prefix_route(app.route, url_root)
    app.config['JSON_AS_ASCII'] = False

    initialize(app, args)

    @app.route('/', methods=['GET'])
    def index():
        return jsonify("hello nlp server")

    CORS(app)

    http_server = WSGIServer((host, port), app)
    logger.info("Model loaded, serving deepsegment on port %d" % port)
    http_server.serve_forever()


if __name__ == '__main__':
    opt = opts.model_opts()
    config = yaml.load(open(opt.config, "r"))
    config = Namespace(**config, **vars(opt))

    device, devices_id = misc_utils.set_cuda(config)
    config.device = device

    # stdout_handler = prepare_global_logging(args.serialization_dir, args.file_friendly_logging)
    start(config, url_root=config.url_root, host=config.ip, port=config.port)
    # cleanup_global_logging(stdout_handler)
