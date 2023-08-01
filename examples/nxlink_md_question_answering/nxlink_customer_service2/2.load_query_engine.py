#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import json
import os
from pathlib import Path

import chromadb
from chromadb.api.models.Collection import Collection
from langchain.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.composability.graph import ComposableGraph
from llama_index.indices.list.base import ListIndex
from llama_index.indices.vector_store.base import VectorStoreIndex
from llama_index.indices.service_context import ServiceContext
from llama_index.llm_predictor.base import LLMPredictor
from llama_index.prompts.default_prompts import DEFAULT_SIMPLE_INPUT_PROMPT, DEFAULT_TEXT_QA_PROMPT
from llama_index.query_engine.graph_query_engine import ComposableGraphQueryEngine
from llama_index.response.schema import Response
from llama_index.response_synthesizers.tree_summarize import TreeSummarize
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from tqdm import tqdm

from project_settings import project_path
import project_settings as settings
from toolbox.llama_index.readers.file.markdown_reader import MarkdownReader
from toolbox.llama_index.indices.markdown_index.base import MarkDownIndex


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--document_dir",
        default=(project_path / "data/NXLink").as_posix(),
        type=str
    )
    parser.add_argument(
        "--persist_dir",
        default=(project_path / "cache/persist_dir/nxlink_customer_service2/").as_posix(),
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

    # chroma
    chroma_client = chromadb.PersistentClient(path=args.persist_dir)
    collection = chroma_client.create_collection(name="nxlink_customer_service2")

    # data
    reader = MarkdownReader()
    document_dir = Path(args.document_dir)

    documents = list()
    for filename in tqdm(document_dir.glob("**/*.md")):
        documents_ = reader.load_data(file=filename)
        documents.extend(documents_)

    with open(os.path.join(args.persist_dir, "index_struct.json"), "r", encoding="utf-8") as f:
        json_str = f.read()
    index_struct = VectorStoreIndex.index_struct_cls.from_json(json_str)

    storage_context: StorageContext = StorageContext.from_defaults(
        vector_store=ChromaVectorStore(
            chroma_collection=collection
        ),
        persist_dir=args.persist_dir
    )

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
    index = VectorStoreIndex(
        index_struct=index_struct,
        storage_context=storage_context,
        service_context=service_context,
        response_synthesizer=response_synthesizer,
    )
    query_engine = index.as_query_engine()

    # query
    while True:
        query = input("Query: ")
        if query == "Quit":
            break
        response: Response = query_engine.query(query)
        print(response.response)
        print(len(response.source_nodes))
        for source_node in response.source_nodes:
            print(source_node.node.text)
            print(source_node.score)
            print("-" * 100)

    return


if __name__ == '__main__':
    main()
