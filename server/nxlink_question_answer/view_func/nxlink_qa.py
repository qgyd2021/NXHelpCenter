#!/usr/bin/python3
# -*- coding: utf-8 -*-
import logging
import os
from typing import List

from flask import render_template, request
import jsonschema
import requests

from server.exception import ExpectedError
from server.flask_server.route_wrap.common_route_wrap import common_route_wrap
from server.nxlink_question_answer.schema import nxlink_qa
from server.nxlink_question_answer.service.nxlink_qa import get_nxlink_qa_instance

from toolbox.logging.misc import json_2_str


logger = logging.getLogger("server")


def nxlink_qa_page():
    return render_template("nxlink_qa.html")


@common_route_wrap
def query_view_func():
    args = request.form
    logger.info("query_view_func, args: {}".format(json_2_str(args)))

    # request body verification
    try:
        jsonschema.validate(args, nxlink_qa.nxlink_qa_request_schema)
    except (jsonschema.exceptions.ValidationError,
            jsonschema.exceptions.SchemaError, ) as e:
        raise ExpectedError(
            status_code=60401,
            message="request body invalid. ",
            detail=str(e)
        )

    query = args['query']
    service = get_nxlink_qa_instance()

    result = service.query(query)

    return result


if __name__ == '__main__':
    pass
