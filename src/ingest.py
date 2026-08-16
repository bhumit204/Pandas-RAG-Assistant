from pathlib import Path
from bs4 import BeautifulSoup
from typing import Any, Dict, Iterable, List
import requests
import zipfile
import os
import glob
import shutil

def sliding_window(
    seq: Iterable[Any],
    size: int,
    step: int,
) -> List[Dict[str, Any]]:
    """Create overlapping chunks from a sequence using a sliding window approach.

    Args:
        seq: The input sequence (string or list) to be chunked.
        size: The size of each chunk/window.
        step: The step size between consecutive windows.

    Returns:
        A list of dictionaries, each containing:
            - 'start': The starting position of the chunk in the original sequence
            - 'content': The chunk content

    Raises:
        ValueError: If size or step are not positive integers.

    Example:
        >>> sliding_window("hello world", size=5, step=3)
        [{'start': 0, 'content': 'hello'}, {'start': 3, 'content': 'lo wo'}]
    """
    if step >= size:
        raise ValueError("step (overlap) must be smaller than size")

    n = len(seq)

    if n <= size:
        return [{"content": seq}]

    chunks = []
    stride = size - step

    for start in range(0, n, stride):
        end = start + size
        chunk = seq[start:end]

        chunks.append({
            "content": chunk
        })

        if end >= n:
            break

    return chunks

def chunk_documents(
    documents: Iterable[Dict[str, str]],
    size: int = 2000,
    step: int = 1000,
    content_field_name: str = "content",
) -> List[Dict[str, str]]:
    """Split a collection of documents into smaller chunks using sliding windows.

    Takes documents and breaks their content into overlapping chunks while preserving
    all other document metadata (filename, etc.) in each chunk.

    Args:
        documents: An iterable of document dictionaries. Each document must have a content field.
        size: The maximum size of each chunk. Defaults to 2000.
        step: The step size between chunks. Defaults to 1000.
        content_field_name: The name of the field containing document content.
            Defaults to 'content'.

    Returns:
        A list of chunk dictionaries. Each chunk contains:
            - All original document fields except the content field
            - 'start': Starting position of the chunk in original content
            - 'content': The chunk content

    Example:
        >>> documents = [{'content': 'long text...', 'filename': 'doc.txt'}]
        >>> chunks = chunk_documents(documents, size=100, step=50)
        >>> # Or with custom content field:
        >>> documents = [{'text': 'long text...', 'filename': 'doc.txt'}]
        >>> chunks = chunk_documents(documents, content_field_name='text')
    """
    results = []

    for doc in documents:
        doc_copy = doc.copy()
        doc_content = doc_copy.pop(content_field_name)
        chunks = sliding_window(doc_content, size=size, step=step)
        for chunk in chunks:
            chunk.update(doc_copy)
        results.extend(chunks)

    return results

def download_unzip_file(url, output_file_zip, output_file):
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(output_file_zip, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

    print(f"Downloaded to {output_file_zip}")

    with zipfile.ZipFile(output_file_zip, "r") as zf:
        zf.extractall(output_file)


    files = glob.glob("..//data//pandas_docs//*.html")

    for file in files:
        os.remove(file)

    shutil.rmtree("..//data//pandas_docs//generated")

def load_data(folder_path):
    folder_path = Path(folder_path)
    # min_num = 100000000
    # max_num = 0
    # sum_num = 0
    # num = 0
    data_lst = []
    for file_path in folder_path.rglob("*.html"):
        print(f'processing:{file_path}')
        with open(file_path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f.read(), "html.parser")
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract clean text
            text = soup.get_text(separator=" ", strip=True)
            if len(text) < 55:
                print(f"text:{text}")
                continue
        data_lst.append({'content':text.strip(),'file_name':str(file_path)[20:]})
        # break
    chunks = chunk_documents(data_lst, size=2000, step=600)
    print(f"Number of chunks: {len(chunks)}")
    print('load_data completed')
    return chunks

if __name__ == "__main__":
    # from dotenv import load_dotenv
    from vector_db import qdrant_insert
    import numpy as np

    url = "https://pandas.pydata.org/docs/pandas.zip"
    output_file_zip = "..//data//pandas.zip"
    output_file = "..//data//pandas_docs"

    # load_dotenv()
    download_unzip_file(url, output_file_zip, output_file)
    data = load_data('..//data//pandas_docs')
    data_array = np.array(data)
    qdrant_insert(data_array)