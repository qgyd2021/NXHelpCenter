#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import logging
import os
import sys
from datetime import datetime, timedelta

pwd = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(pwd, '../../'))

from flask import Flask
from gevent import pywsgi

from server import log
from server.nxlink_question_answer import settings

log.setup(log_directory=settings.log_directory)

from server.flask_server.view_func.heart_beat import heart_beat
from server.nxlink_question_answer.view_func import nxlink_qa

logger = logging.getLogger('server')


flask_app = Flask(
    __name__,
    static_url_path='/',
    static_folder='static',
    template_folder='static/templates',
)

flask_app.add_url_rule(rule="/HeartBeat", view_func=heart_beat, methods=["GET", "POST"], endpoint="HeartBeat")
flask_app.add_url_rule(rule="/NXLinkQA", view_func=nxlink_qa.nxlink_qa_page, methods=["GET"], endpoint="NXLinkQAPage")
flask_app.add_url_rule(rule="/NXLinkQA/query", view_func=nxlink_qa.query_view_func, methods=["POST"], endpoint="NXLinkQAQuery")

# http://10.75.27.247:12023/NXLinkQA
# http://127.0.0.1:12023/NXLinkQA

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port',
        default=settings.port,
        type=int,
    )
    args = parser.parse_args()

    logger.info('model server is already, 127.0.0.1:{}'.format(args.port))

    # flask_app.run(
    #     host='0.0.0.0',
    #     port=args.port,
    # )

    server = pywsgi.WSGIServer(
        listener=('0.0.0.0', args.port),
        application=flask_app
    )
    server.serve_forever()
