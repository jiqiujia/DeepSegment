# -*- encoding: utf-8 -*-
'''
 Created on 2019/2/21.
 @author: eddielin
'''
from flask import Flask, jsonify, request

import logging
from .server import Server
from .response import Response, Status

ROOT_URL = '/deepsegment'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def url(sub_url):
    return ROOT_URL + sub_url


def router(app: Flask, server: Server):
    @app.route(url('/batch'), methods=['POST'])
    def batch_deep_segment():
        payload = request.get_json(force=True)
        logger.info("reqeust %s" % payload)
        response = Response()
        try:
            data = server.batch_predict_line(**payload)
            response.data = data
            response.status = Status.OK
        except Exception as ex:
            response.status = Status.ERROR
            response.msg = str(ex)

        return jsonify(response.to_dict())