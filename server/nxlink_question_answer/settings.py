#!/usr/bin/python3
# -*- coding: utf-8 -*-
import os
from typing import List

from project_settings import project_path
from toolbox.os.environment import EnvironmentManager

log_directory = os.path.join(project_path, "server/nxlink_question_answer/logs")
os.makedirs(log_directory, exist_ok=True)

environment = EnvironmentManager(
    path=os.path.join(project_path, "server/nxlink_question_answer/dotenv"),
    env=os.environ.get("environment", "dev"),
)

port = environment.get(key="port", default=12023, dtype=int)


nxlink_question_answer_dataset = environment.get(
    key="nxlink_question_answer_dataset",
    default=(project_path / "data/nxlink_question_answer").as_posix(),
    dtype=str
)
nxlink_faq_filename = environment.get(
    key="nxlink_faq_filename",
    default="nxlink_faq.jsonl",
    dtype=str
)

elastic_host = environment.get(key="elastic_host", default="127.0.0.1", dtype=str)
elastic_port = environment.get(key="elastic_port", default=9200, dtype=int)
elastic_index = environment.get(key="elastic_index", default="nxlink_elasticsearch_retrieval_index", dtype=str)
elastic_query_top_k = environment.get(key="elastic_query_top_k", default=5, dtype=int)


faq_prefix_prompt_str = """
你是一个问答机器人, 用户给定一个问题, 我们会从数据库检索出一些与该问题可能相关的问答对. 
这些问答对可能与用户问题有关也可能无关, 你需要根据这些问答对中的信息来回答用户的问题. 

Tips: 
1. 注意 ExampleAnswer 与 ExampleQuestion 是成对的问答, 你需要自己判断哪些问答是有帮助的. 
2. 有可能全部问答都是无用的, 那么你应该回答不知道. 
3. 不要编造答案, 对答案没有很强的信心时, 请直接回答不知道. 

=========

ExampleQuestion: 一个手机号码最多能注册多少个 Facebook 账号？
ExampleAnswer: 一个手机号码最多只能注册一个 Facebook 账号。

ExampleQuestion: 什么是Facebook官方渠道？
ExampleAnswer: Facebook官方渠道是Facebook官方提供的渠道，用户可以在这些渠道上免费创建Facebook个人账号。

ExampleQuestion: 什么是Facebook官方颁发的身份证件？
ExampleAnswer: Facebook官方颁发的身份证件是政府机构颁发的身份证件，例如身份证、护照或驾驶执照。

ExampleQuestion: 怎样注册 Facebook 账号？
ExampleAnswer: 打开Facebook的官网注册账号（https://www.facebook.com），填写基本信息，验证邮箱/手机号码，验证通过即注册成功。

回答此用户问题. 
UserQuestion: facebook注册官网是多少
Answer: https://www.facebook.com

=========
"""

faq_prefix_prompt_str = environment.get(key="llm_prompt_system", default=faq_prefix_prompt_str, dtype=str)

fap_example_prompt_str = """
ExampleQuestion: {question}
ExampleAnswer: {answer}
"""
fap_example_prompt_str = environment.get(key="llm_prompt_faq_examples", default=fap_example_prompt_str, dtype=str)


faq_suffix_prompt_str = """
回答此用户问题. 
UserQuestion: {user_question}
Answer: 
"""
faq_suffix_prompt_str = environment.get(key="llm_prompt_qa_example", default=faq_suffix_prompt_str, dtype=str)


openai_api_key = environment.get(
    key="openai_api_key",
    default=environment.get("openai_api_key", default=None, dtype=str),
    dtype=str
)


if __name__ == '__main__':
    pass
