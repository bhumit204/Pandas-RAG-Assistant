
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from embedder import Embedder
from reranker import Reranker
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

load_dotenv()
embed = Embedder('models\\Xenova\\all-MiniLM-L6-v2')
QDRANT_COLLECTION = "enterprise_rag"

# Initialize Qdrant Client
client = QdrantClient(
    url=os.getenv("QDRANT_CLUSTER_ENDPOINT"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=3600,
)

BATCH_SIZE = 200
MAX_WORKERS = 4

def upload_batch(points, batch_no):
    print(f"Uploading batch {batch_no} ({len(points)} points)...")

    client.upsert(
        collection_name=QDRANT_COLLECTION,
        points=points,
        wait=True
    )

    print(f"Batch {batch_no} uploaded")


def qdrant_insert(chunks):

    # Create collection if it doesn't exist
    if not client.collection_exists(QDRANT_COLLECTION):
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            ),
        )

    # Create batches
    batches = []

    for indx in range(0, len(chunks), BATCH_SIZE):

        print(f"Processing chunks: {indx} -> {indx + BATCH_SIZE}")

        data = chunks[indx:indx + BATCH_SIZE]

        # Generate embeddings for this batch
        embeddings = embed.encode_batch(
            [chunk["content"] for chunk in data]
        )

        points = []

        for offset, (chunk, vector) in enumerate(
            zip(data, embeddings)
        ):

            # Global unique ID
            point_id = indx + offset + 1

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector.tolist()
                    if hasattr(vector, "tolist")
                    else vector,
                    payload={
                        "text": chunk["content"],
                        "source": chunk["file_name"],
                    },
                )
            )

        batches.append((points, indx // BATCH_SIZE + 1))

    # Parallel Qdrant uploads
    print(f"Uploading {len(batches)} batches using {MAX_WORKERS} workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:

        futures = [
            executor.submit(upload_batch, points, batch_no)
            for points, batch_no in batches
        ]

        for future in as_completed(futures):
            # Raises exception if an upload failed
            future.result()

    print(f"Successfully uploaded {len(chunks)} points")


def search(query, limit=30):
    from datetime import datetime

    print(f"search started {datetime.now()}")

    embedding = embed.encode(query)

    results = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=embedding.tolist(),
        limit=limit
    ).points
    
    print(f"Rerank started {datetime.now()}")
    reranker = Reranker()
    reranked = reranker.rerank(
    query,
    results
)
    print(f"search Ended {datetime.now()}")
    return [
        {
            "content": r.payload["text"],
            "file_name": r.payload["source"],
            "score": r.score,
        }
        for r in reranked
    ]

def delete_collection(q_collection):
    if client.collection_exists(q_collection):
        client.delete_collection(q_collection)
    else:
        print(f"{q_collection} doesn't exist")
