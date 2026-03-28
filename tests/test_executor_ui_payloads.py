from executor import (
    extract_source_nodes,
    format_exception_result,
    format_operation_result,
    format_query_result,
)


def test_format_query_result_includes_answer_sources_and_debug():
    payload = format_query_result(
        answer="曹操曾迎天子以令诸侯。",
        sources=[
            {
                "title": "三国志",
                "content": "太祖迎天子都许。",
                "score": 0.91,
                "rank": 1,
                "file_name": "baihuasanguozhi.txt",
            }
        ],
        debug={"mode": "milvus", "retrieved_count": 5, "used_count": 3},
    )

    assert payload["ok"] is True
    assert payload["answer"] == "曹操曾迎天子以令诸侯。"
    assert payload["sources"][0]["title"] == "三国志"
    assert payload["debug"]["retrieved_count"] == 5
    assert payload["error"] is None


def test_format_operation_result_returns_error_payload():
    payload = format_operation_result(False, "build failed", error="missing path")

    assert payload == {
        "ok": False,
        "status": "build failed",
        "error": "missing path",
        "details": None,
    }


class DummyNode:
    def __init__(self, text, metadata=None, score=0.5):
        self.score = score
        self.metadata = metadata or {}
        self._text = text

    def get_content(self, metadata_mode=None):
        return self._text


class DummyContext:
    def __init__(self, node):
        self.node = node
        self.score = node.score


def test_extract_source_nodes_normalizes_missing_metadata():
    contexts = [
        DummyContext(
            DummyNode(
                "原文片段",
                metadata={"file_name": "baihuasanguozhi.txt", "digest_from": "三国志"},
                score=0.88,
            )
        )
    ]

    sources = extract_source_nodes(contexts)

    assert sources == [
        {
            "title": "三国志",
            "content": "原文片段",
            "score": 0.88,
            "rank": 1,
            "file_name": "baihuasanguozhi.txt",
        }
    ]


def test_format_exception_result_returns_error_shape():
    payload = format_exception_result("query failed", RuntimeError("boom"))

    assert payload["ok"] is False
    assert payload["status"] == "query failed"
    assert payload["error"] == "boom"
