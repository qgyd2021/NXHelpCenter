#!/usr/bin/python3
# -*- coding: utf-8 -*-
import json
import os
from typing import List

import elasticsearch as es
from elasticsearch import Elasticsearch, helpers
import jieba
from langchain.chains.llm import LLMChain
from langchain.llms import OpenAI, HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.prompts.few_shot import FewShotPromptTemplate

from server.nxlink_question_answer import settings


example_prompt = PromptTemplate.from_template(settings.fap_example_prompt_str)


def get_faq_prompt_template(examples: List[dict]):
    prompt = FewShotPromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
        prefix=settings.faq_prefix_prompt_str,
        suffix=settings.faq_suffix_prompt_str,
        example_separator="",
        input_variables=["user_question"]
    )
    return prompt


class NXLinkFAQElasticIndex(object):

    mapping = {
        "properties": {
            "question": {
                "type": "text",
                "analyzer": "whitespace",
                "search_analyzer": "whitespace"
            },
            "question_preprocessed": {
                "type": "text",
                "analyzer": "whitespace",
                "search_analyzer": "whitespace"
            },
            "answer": {
                "type": "text",
            },
            "filename": {
                "type": "keyword"
            },
            "header": {
                "type": "keyword",
            },
            "product": {
                "type": "keyword"
            }
        }
    }

    def __init__(self,
                 nxlink_faq_file: str,
                 elastic_host: str,
                 elastic_port: int,
                 elastic_index: str,
                 elastic_query_top_k: int = 5
                 ):
        self.nxlink_faq_file = nxlink_faq_file
        self.elastic_host = elastic_host
        self.elastic_port = elastic_port
        self.elastic_index = elastic_index
        self.elastic_query_top_k = elastic_query_top_k

        self.es_client = es.Elasticsearch(
            hosts=[self.elastic_host],
            port=self.elastic_port
        )
        ping_flag: bool = self.es_client.ping()
        if not ping_flag:
            raise AssertionError("elasticsearch ping failed.")

    def _build_elastic_index(self):
        if self.es_client.indices.exists(index=self.elastic_index):
            self.es_client.indices.delete(index=self.elastic_index)

        # 设置索引最大数量据.
        body = {
            "settings": {
                "index": {
                    "max_result_window": 100000
                }
            }
        }
        self.es_client.indices.create(index=self.elastic_index, body=body)

        # 设置文档结构
        self.es_client.indices.put_mapping(
            index=self.elastic_index,
            doc_type='_doc',
            body=self.mapping,
            params={"include_type_name": "true"}
        )

        # 写入新的数据
        rows = list()
        with open(self.nxlink_faq_file, "r", encoding="utf-8") as f:
            for row in f:
                row = json.loads(row)
                filename = row["filename"]
                faq: List[dict] = row["faq"]

                for qa in faq:
                    section = qa["section"]
                    question = qa["standard_question"]
                    answer = qa["answer"]

                    question_preprocessed = self.text_split(question)

                    row = {
                        "question": question,
                        "question_preprocessed": question_preprocessed,
                        "answer": answer,
                        "filename": filename,
                        "header": section,
                        "product": "nxlink",

                    }
                    rows.append({
                        '_op_type': 'index',
                        '_index': self.elastic_index,
                        '_source': row
                    })
            helpers.bulk(client=self.es_client, actions=rows)

            # 刷新数据
            self.es_client.indices.refresh(index=self.elastic_index)

        return

    def text_split(self, text: str):
        text = str(text).lower()
        tokens = jieba.lcut(text)
        tokens = [token for token in tokens if not len(token.strip()) == 0]
        return tokens

    def query(self, query: str):
        tokens = self.text_split(query)
        query_preprocessed = " ".join(tokens)

        query = {
            "query": {
                "bool": {
                    "must": [{
                        "match": {
                            "question_preprocessed": query_preprocessed
                        }
                    }],
                    "filter": [
                        {"term": {"product": "nxlink"}},
                    ]
                },
            },
        }
        js = self.es_client.search(
            index=self.elastic_index,
            size=self.elastic_query_top_k,
            body=query
        )

        hits = js["hits"]["hits"]

        result = list()
        for hit in hits:
            score = hit['_score']
            source = hit['_source']

            question = source["question"]
            question_preprocessed = source["question_preprocessed"]
            answer = source["answer"]
            filename = source["filename"]
            header = source["header"]
            product = source["product"]

            result.append({
                "score": score,
                "question": question,
                "answer": answer,
                "filename": filename,
                "header": header,
                "product": product,
            })

        return result


class NXLinkQA(object):
    def __init__(self,
                 faq_elastic_index: NXLinkFAQElasticIndex,
                 openai_api_key: str
                 ):
        self.faq_elastic_index = faq_elastic_index
        self.openai_api_key = openai_api_key

        self.llm = OpenAI(
            temperature=0.7,
            max_tokens=1024,
            n=10,
            openai_api_key=self.openai_api_key
        )

    def query(self, query: str):
        faq_recall = self.faq_elastic_index.query(query)
        examples = [{"question": o["question"], "answer": o["answer"]} for o in faq_recall]

        prompt = get_faq_prompt_template(examples)

        llm_chain = LLMChain(llm=self.llm, prompt=prompt)

        answer = llm_chain.predict(user_question=query)

        result = {
            "answer": answer,
            "faq_recall": faq_recall,
        }
        return result


_nxlink_qa_service: NXLinkQA = None


def get_nxlink_qa_instance():
    global _nxlink_qa_service

    if _nxlink_qa_service is None:
        _nxlink_qa_service = NXLinkQA(
            faq_elastic_index=NXLinkFAQElasticIndex(
                nxlink_faq_file=os.path.join(settings.nxlink_question_answer_dataset, settings.nxlink_faq_filename),
                elastic_host=settings.elastic_host,
                elastic_port=settings.elastic_port,
                elastic_index=settings.elastic_index,
                elastic_query_top_k=settings.elastic_query_top_k,
            ),
            openai_api_key=settings.openai_api_key
        )

    return _nxlink_qa_service


if __name__ == '__main__':
    pass
