from executor import format_operation_result, format_query_result


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
