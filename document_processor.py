import os
import hashlib
from pathlib import Path
from typing import List
from docling.document_converter import DocumentConverter
from langchain_text_splitters import MarkdownHeaderTextSplitter

class DocumentProcessor:
    def __init__(self):
        self.headers = [("#", "Header 1"), ("##", "Header 2")]

    def process(self, files: List) -> List:
        """Processes documents using Docling and splits them into structured chunks."""
        all_chunks = []
        seen_hashes = set()
        
        for file in files:
            if not file.name.endswith(('.pdf', '.docx', '.txt', '.md')):
                continue
                
            converter = DocumentConverter()
            result = converter.convert(file.name)
            markdown_text = result.document.export_to_markdown()
            
            splitter = MarkdownHeaderTextSplitter(self.headers)
            chunks = splitter.split_text(markdown_text)
            
            for chunk in chunks:
                chunk_hash = hashlib.sha256(chunk.page_content.encode()).hexdigest()
                if chunk_hash not in seen_hashes:
                    all_chunks.append(chunk)
                    seen_hashes.add(chunk_hash)
                    
        return all_chunks
