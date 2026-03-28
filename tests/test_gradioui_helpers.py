from gradioui import (
    ensure_ready_state,
    map_operation_payload_to_status,
    map_query_payload_to_outputs,
    render_sources_markdown,
)


def test_ensure_ready_state_rejects_missing_executor():
    ok, message = ensure_ready_state({"executor": None, "initialized": False})

    assert ok is False
    assert "请先初始化" in message


def test_render_sources_markdown_outputs_ranked_cards():
    markdown = render_sources_markdown(
        [
            {
                "title": "三国志",
                "content": "太祖迎天子都许。",
                "score": 0.91,
                "rank": 1,
                "file_name": "baihuasanguozhi.txt",
            }
        ]
    )

    assert "证据 1" in markdown
    assert "三国志" in markdown
    assert "太祖迎天子都许。" in markdown


def test_map_query_payload_to_outputs_returns_answer_status_and_sources():
    payload = {
        "ok": True,
        "status": "success",
        "answer": "回答内容",
        "sources": [
            {
                "title": "三国志",
                "content": "原文",
                "rank": 1,
                "score": 0.9,
                "file_name": "a.txt",
            }
        ],
        "debug": {"retrieved_count": 3},
        "error": None,
    }

    answer, status, sources_md, debug_md = map_query_payload_to_outputs(payload)

    assert answer == "回答内容"
    assert status == "success"
    assert "三国志" in sources_md
    assert "retrieved_count" in debug_md


def test_map_operation_payload_to_status_prefers_error_message():
    status = map_operation_payload_to_status(
        {"ok": False, "status": "delete failed", "error": "missing file", "details": None}
    )

    assert status == "delete failed: missing file"
