from flashrank import Ranker, RerankRequest


class Reranker:

    def __init__(self):
        self.ranker = Ranker(model_name="ms-marco-TinyBERT-L-2-v2")

    def rerank(self, query, search_results, top_k=5):

        passages = []

        for idx, result in enumerate(search_results):
            passages.append(
                {
                    "id": idx,
                    "text": result.payload["text"],
                    "meta": result,
                }
            )

        request = RerankRequest(
            query=query,
            passages=passages,
        )

        ranked = self.ranker.rerank(request)

        return [
            item["meta"]
            for item in ranked[:top_k]
        ]