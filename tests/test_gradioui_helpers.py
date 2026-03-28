from gradioui import ensure_ready_state, render_sources_markdown


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
