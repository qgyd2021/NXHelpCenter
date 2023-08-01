#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path

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
        "--openai_api_key",
        default=settings.environment.get("openai_api_key", default=None, dtype=str),
        type=str
    )
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    reader = MarkdownReader()

    document_dir = Path(args.document_dir)

    children_indices = list()
    index_summaries = list()
    custom_query_engines = dict()
    for filename in tqdm(document_dir.glob("**/*.md")):
        documents = reader.load_data(file=filename)

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
        children_indices.append(index)
        index_summaries.append(index.summary)
        custom_query_engines[index.index_id] = index.as_query_engine(similarity_top_k=1)

    storage_context: StorageContext = StorageContext.from_defaults()
    service_context: ServiceContext = ServiceContext.from_defaults(
        llm_predictor=LLMPredictor(
            llm=OpenAI(
                openai_api_key=args.openai_api_key
            )
        ),
        embed_model=OpenAIEmbedding(api_key=args.openai_api_key),
    )
    # composable graph
    graph = ComposableGraph.from_indices(
        VectorStoreIndex,
        children_indices=children_indices,
        index_summaries=index_summaries,
        service_context=service_context,
        storage_context=storage_context,
    )
    custom_query_engines[graph.root_id] = graph.root_index.as_query_engine(similarity_top_k=1)

    query_engine = ComposableGraphQueryEngine(
        graph=graph,
        custom_query_engines=custom_query_engines,
        recursive=True
    )

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
