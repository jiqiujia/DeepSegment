# -*- encoding: utf-8 -*-
'''
 Created on 2019/2/21.
 @author: eddielin
'''
import json

from flask import request, jsonify


class Status(object):
    OK = 'ok'
    ERROR = "error"


class Response(object):

    def __init__(self, data='', status='', msg=''):
        self.data = data
        self.status = status
        self.msg = msg

    def to_dict(self):
        return dict(
            data=self.data,
            status=self.status,
            msg=self.msg
        )

    def __str__(self) -> str:
        return json.dumps(self.to_dict())


def decorator(func):
    payload = request.get_json(force=True)
    response = Response()
    try:
        data = func(**payload)
        response.data = data
        response.status = Status.OK
    except Exception as ex:
        response.status = Status.ERROR
        response.msg = str(ex)

    return jsonify(response)