#!/usr/bin/python3
# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import Any, Dict, Generic, List, Optional, Sequence, Type, TypeVar

from llama_index.data_structs.data_structs import IndexDict
from llama_index.indices.base import BaseIndex, IndexType
from llama_index.indices.base_retriever import BaseRetriever
from llama_index.indices.service_context import ServiceContext
from llama_index.langchain_helpers.text_splitter import TokenTextSplitter
from llama_index.node_parser.simple import SimpleNodeParser
from llama_index.response.schema import Response
from llama_index.response_synthesizers import BaseSynthesizer, get_response_synthesizer
from llama_index.schema import (
    BaseNode, Document, ImageNode, IndexNode, MetadataMode,
    NodeRelationship, NodeWithScore,
    RelatedNodeInfo, TextNode
)

from llama_index.storage.docstore.types import RefDocInfo
from llama_index.storage.storage_context import StorageContext
from llama_index.vector_stores.types import NodeWithEmbedding, VectorStore
from tqdm import tqdm

from toolbox.llama_index.schema import EmbedTextNode


DEFAULT_SUMMARY_QUERY = (
    "Give a concise summary of this document. Also describe some of the questions "
    "that this document can answer. "
)


class MarkDownIndex(BaseIndex[IndexDict]):
    """
    markdown 索引.

    数据结构:
    (1)一个索引库就是一个 Markdown 文件.
    (2)每个 head 标签做为一个 Document. 对 Document 进行 Summary 以得到 TextNode.
    (3)当一个 TextNode 被检索到时, 将其对应的 Document 完整内容用于执行 Summary 回答.
    (4)与 MarkdownReader 配合使用. 其中 metadata 中 header=-1 的执行 text chunk 分割, 其它的进行 summary 总结.

    检索方法:
    (1)对各单元进行 summary 然后执行 embedding 向量召回.

    """

    index_struct_cls = IndexDict

    def __init__(
        self,
        nodes: Optional[Sequence[BaseNode]] = None,
        index_struct: Optional[IndexDict] = None,
        service_context: Optional[ServiceContext] = None,
        storage_context: Optional[StorageContext] = None,
        store_nodes_override: bool = False,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        self._store_nodes_override = store_nodes_override
        super().__init__(
            nodes=nodes,
            index_struct=index_struct,
            service_context=service_context,
            storage_context=storage_context,
            show_progress=show_progress,
            **kwargs,
        )

    def _get_node_embedding_results(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> List[NodeWithEmbedding]:
        id_to_embed_map: Dict[str, List[float]] = {}

        for n in nodes:
            if n.embedding is None:
                self._service_context.embed_model.queue_text_for_embedding(
                    n.node_id, n.get_content(metadata_mode=MetadataMode.EMBED)
                )
            else:
                id_to_embed_map[n.node_id] = n.embedding

        # call embedding model to get embeddings
        (
            result_ids,
            result_embeddings,
        ) = self._service_context.embed_model.get_queued_text_embeddings(show_progress)
        for new_id, text_embedding in zip(result_ids, result_embeddings):
            id_to_embed_map[new_id] = text_embedding

        results = []
        for node in nodes:
            embedding = id_to_embed_map[node.node_id]
            result = NodeWithEmbedding(node=node, embedding=embedding)
            results.append(result)
        return results

    def _add_nodes_to_index(
        self,
        index_struct: IndexDict,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
    ) -> None:
        if not nodes:
            return

        embedding_results = self._get_node_embedding_results(nodes, show_progress)
        new_ids = self._vector_store.add(embedding_results)

        if not self._vector_store.stores_text or self._store_nodes_override:
            # NOTE: if the vector store doesn't store text,
            # we need to add the nodes to the index struct and document store
            for result, new_id in zip(embedding_results, new_ids):
                index_struct.add_node(result.node, text_id=new_id)
                self._docstore.add_documents([result.node], allow_update=True)
        else:
            # NOTE: if the vector store keeps text,
            # we only need to add image and index nodes
            for result, new_id in zip(embedding_results, new_ids):
                if isinstance(result.node, (ImageNode, IndexNode)):
                    index_struct.add_node(result.node, text_id=new_id)
                    self._docstore.add_documents([result.node], allow_update=True)

    def _build_index_from_nodes(self, nodes: Sequence[BaseNode]) -> IndexDict:
        index_struct = self.index_struct_cls()
        self._add_nodes_to_index(
            index_struct, nodes, show_progress=self._show_progress
        )
        return index_struct

    @classmethod
    def from_documents(
        cls: Type[IndexType],
        documents: Sequence[Document],
        storage_context: Optional[StorageContext] = None,
        service_context: Optional[ServiceContext] = None,
        response_synthesizer: Optional[BaseSynthesizer] = None,
        summary_query: str = DEFAULT_SUMMARY_QUERY,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> IndexType:

        storage_context = storage_context or StorageContext.from_defaults()
        service_context = service_context or ServiceContext.from_defaults()
        docstore = storage_context.docstore

        with service_context.callback_manager.as_trace("index_construction"):
            for doc in documents:
                docstore.set_document_hash(doc.get_doc_id(), doc.hash)

            nodes = list()
            for document in tqdm(documents):
                if document.metadata["header"] == -1:
                    nodes_: List[TextNode] = service_context.node_parser.get_nodes_from_documents(
                        [document], show_progress=show_progress
                    )
                    for node_ in nodes_:
                        node = EmbedTextNode(
                            text=node_.text,
                            embed_text=node_.text,
                            embedding=node_.embedding,
                            metadata=node_.metadata,
                            excluded_embed_metadata_keys=node_.excluded_embed_metadata_keys,
                            excluded_llm_metadata_keys=node_.excluded_llm_metadata_keys,
                            metadata_seperator=node_.metadata_seperator,
                            text_template=node_.text_template,
                            relationships=node_.relationships,
                        )
                        nodes.append(node)
                else:
                    summary_response: Response = response_synthesizer.synthesize(
                        query=summary_query,
                        nodes=[NodeWithScore(node=document)],
                    )
                    node = EmbedTextNode(
                        text=document.text,
                        embed_text=summary_response.response,
                        embedding=document.embedding,
                        metadata=document.metadata,
                        excluded_embed_metadata_keys=document.excluded_embed_metadata_keys,
                        excluded_llm_metadata_keys=document.excluded_llm_metadata_keys,
                        metadata_seperator=document.metadata_seperator,
                        text_template=document.text_template,
                        relationships={
                            NodeRelationship.SOURCE: document.as_related_node_info()
                        },
                    )
                    nodes.append(node)

        return cls(
            nodes=nodes,
            storage_context=storage_context,
            service_context=service_context,
            show_progress=show_progress,
            **kwargs,
        )

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        pass

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        pass

    def as_retriever(self, **kwargs: Any) -> BaseRetriever:
        # NOTE: lazy import
        from llama_index.indices.vector_store.retrievers import VectorIndexRetriever

        return VectorIndexRetriever(
            self,
            **kwargs,
        )

    @property
    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        raise NotImplementedError

    @property
    def vector_store(self) -> VectorStore:
        return self._vector_store
