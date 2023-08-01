#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
from typing import Dict, List

from langchain.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.postprocessor import SimilarityPostprocessor
from llama_index.indices.service_context import ServiceContext
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.prompts.default_prompts import DEFAULT_SIMPLE_INPUT_PROMPT, DEFAULT_TEXT_QA_PROMPT
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.response.schema import Response
from llama_index.response_synthesizers.tree_summarize import TreeSummarize
from llama_index.schema import Document
from llama_index.storage.storage_context import StorageContext

from project_settings import project_path
import project_settings as settings
from toolbox.llama_index.readers.file.markdown_reader import MarkdownReader
from toolbox.llama_index.indices.markdown_index.base import MarkDownIndex


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filename",
        default=(project_path / "data/NXLink/1、新手入门/5、嵌入式注册&认证指南(NXLink).md").as_posix(),
        type=str
    )
    parser.add_argument(
        "--openai_api_key",
        default=settings.environment.get("openai_api_key", default=None, dtype=str),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # document
    reader = MarkdownReader()
    documents = reader.load_data(file=Path(args.filename))
    # print(documents)

    storage_context: StorageContext = StorageContext.from_defaults()
    service_context: ServiceContext = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(
            llm=OpenAI(
                openai_api_key=args.openai_api_key
            )
        ),
        embed_model=OpenAIEmbedding(api_key=args.openai_api_key),
    )
    response_synthesizer = TreeSummarize(
        service_context=service_context,
        text_qa_template=DEFAULT_TEXT_QA_PROMPT,
        streaming=False,
    )
    index: MarkDownIndex = MarkDownIndex.from_documents(
        documents,
        storage_context=storage_context,
        service_context=service_context,
        response_synthesizer=response_synthesizer,
    )
    print(index)
    for _, node in index.docstore.docs.items():
        print(node)
        print("-" * 100)

    retriever = index.as_retriever(similarity_top_k=1)
    # result: List[NodeWithScore] = retriever.retrieve(args.query)
    # print(result)
    # print(type(result[0].node))

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer,
        node_postprocessors=[
            SimilarityPostprocessor(similarity_cutoff=0.1)
        ]
    )

    while True:
        query = input("query: ")
        if query == "Quit":
            break
        response: Response = query_engine.query(query)
        print(response.response)
        print(response.source_nodes)

    return


if __name__ == '__main__':
    main()
