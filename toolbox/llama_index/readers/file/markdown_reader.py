#!/usr/bin/python3
# -*- coding: utf-8 -*-
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, cast

from llama_index.readers.base import BaseReader
from llama_index.schema import Document, NodeRelationship, RelatedNodeInfo, TextNode


class MarkdownReader(BaseReader):

    """
    数据结构:
    (1)Markdown 文件下的每一个 head 下的内容被认为是一个完整的信息部分.
    (2)将 Markdown 文件下的每一个 head 下的所属内容保存为一个 Document.

    检索方法:
    (1)建议的用法是, 将每一个 Document 进行 Summary, 之后得到一个 TextNode 作为检索单位.
    (2)当一个 TextNode 匹配之后, 则调用其完整的 text 文档内容作为回答的上下文.
    (3)有些 head 下的内容可能特别长, 将 Markdown 的完整内容进行 text_chunk 分割, 以提供细节部分的问答.

    备注:
    (1)此方法存在冗余, 但其好处是省去了查找 child 子节点再将其整合到一起的过程. 提高了检索速度.
    (2)整个数据结构的设计思路变得简单.

    """

    def __init__(
        self,
        *args: Any,
        remove_hyperlinks: bool = True,
        remove_images: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._remove_hyperlinks = remove_hyperlinks
        self._remove_images = remove_images

    def markdown_to_tups(self, markdown_text: str) -> List[Tuple[Optional[str], str]]:
        markdown_tups: List[Tuple[Optional[str], str]] = []
        lines = markdown_text.split("\n")

        current_header_level = None
        current_header = None
        current_text = ""

        for line in lines:
            if len(str(line).strip()) == 0:
                continue
            header_match = re.match(r"^#+\s", line)
            if header_match:
                if current_header is not None:
                    markdown_tups.append((current_header_level, current_header, current_text))

                head_str = header_match.group().rstrip()
                current_header_level = len(head_str)
                current_header = line
                current_text = ""
            else:
                current_text += line + "\n"
        markdown_tups.append((current_header_level, current_header, current_text))
        return markdown_tups

    def remove_images(self, content: str) -> str:
        pattern = r"!{1}\[\[(.*)\]\]"
        content = re.sub(pattern, "", content)
        return content

    def remove_hyperlinks(self, content: str) -> str:
        pattern = r"\[(.*?)\]\((.*?)\)"
        content = re.sub(pattern, r"\1", content)
        return content

    def _init_parser(self) -> Dict:
        return {}

    def parse_tups(
        self, filepath: Path, errors: str = "ignore"
    ) -> List[Tuple[Optional[str], str]]:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        if self._remove_hyperlinks:
            content = self.remove_hyperlinks(content)
        if self._remove_images:
            content = self.remove_images(content)
        markdown_tups = self.markdown_to_tups(content)
        return markdown_tups

    def load_data(
        self, file: Path, extra_info: Optional[Dict] = None
    ) -> List[Document]:
        extra_info = extra_info or dict()

        tups = self.parse_tups(file)

        doc_idx = 0
        documents = list()
        for header_level, header, text in tups:
            metadata = {
                "filename": file.as_posix(),
                "doc_idx": doc_idx,
                "header": header,
                "header_level": header_level,
                # "parent_headers": list(),
            }
            extra_info.update(metadata)

            if header is not None:
                text = f"{header}\n{text}"
            document = Document(
                text=text,
                metadata=extra_info,
                excluded_embed_metadata_keys=["filename", "doc_idx", "header_level"],
                excluded_llm_metadata_keys=["filename", "doc_idx", "header_level"],
            )
            documents.append(document)
            doc_idx += 1

        l = len(documents)

        # full document
        text = "\n".join([d.text for d in documents])
        metadata = {
            "filename": file.as_posix(),
            "doc_idx": -1,
            "header": -1,
            "header_level": -1,
            # "parent_headers": list(),
        }
        extra_info.update(metadata)
        document = Document(
            text=text,
            metadata=extra_info,
            excluded_embed_metadata_keys=["filename", "doc_idx", "header_level"],
            excluded_llm_metadata_keys=["filename", "doc_idx", "header_level"],
        )

        # merge
        for i in range(l):
            d1 = documents[i]
            header_level1 = d1.metadata["header_level"]

            child_docs = list()
            for j in range(i + 1, l):
                d2 = documents[j]
                header_level2 = d2.metadata["header_level"]
                if header_level2 <= header_level1:
                    break
                # d2.metadata["parent_headers"].append(d1.metadata["header"])
                child_docs.append(d2)

            child_docs = list(sorted(child_docs, key=lambda x: x.metadata["doc_idx"]))
            text = "\n".join([d.text for d in [d1] + child_docs])
            d1.text = text

        result = [document] + documents
        return result


def demo1():
    from llama_index.schema import MetadataMode
    from project_settings import project_path

    reader = MarkdownReader()

    filename = project_path / "data/NXLink/1、新手入门/5、嵌入式注册&认证指南(NXLink).md"
    documents = reader.load_data(file=Path(filename))
    for document in documents:
        # print(document)
        # print(document.get_content(metadata_mode=MetadataMode.LLM))
        print(document.get_content(metadata_mode=MetadataMode.EMBED))
        # print(document.text)
        print("-" * 150)

    return


if __name__ == '__main__':
    demo1()
