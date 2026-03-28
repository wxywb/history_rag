import json

import gradio as gr

from cli import check_required_env, read_yaml_config
from executor import MilvusExecutor, PipelineExecutor

resolutions = ["milvus", "pipeline"]


def default_app_state():
    return {
        "executor": None,
        "mode": None,
        "config_path": "cfgs/config.yaml",
        "initialized": False,
        "status": "未初始化",
    }


def ensure_ready_state(state):
    if not state or not state.get("initialized") or state.get("executor") is None:
        return False, "请先初始化系统后再执行该操作。"
    return True, ""


def render_sources_markdown(sources):
    if not sources:
        return "暂无证据卡片。"

    blocks = []
    for source in sources:
        blocks.append(
            "\n".join(
                [
                    f"### 证据 {source['rank']}: {source['title']}",
                    f"`文件`: {source.get('file_name') or '未知'}",
                    f"`分数`: {source.get('score') if source.get('score') is not None else 'N/A'}",
                    "",
                    source["content"],
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def render_debug_markdown(debug_payload):
    if not debug_payload:
        return ""
    return "```json\n" + json.dumps(debug_payload, ensure_ascii=False, indent=2) + "\n```"


def map_query_payload_to_outputs(payload):
    if not payload["ok"]:
        return "", payload["status"], "暂无证据卡片。", payload.get("error") or ""
    return (
        payload["answer"],
        payload["status"],
        render_sources_markdown(payload["sources"]),
        render_debug_markdown(payload.get("debug")),
    )


def map_operation_payload_to_status(payload):
    if payload["ok"]:
        if payload.get("details"):
            details = payload["details"]
            deleted_count = details.get("deleted_count")
            remaining_count = details.get("remaining_count")
            if deleted_count is not None and remaining_count is not None:
                return f"{payload['status']}，删除 {deleted_count} 条，剩余 {remaining_count} 条。"
        return payload["status"]
    if payload.get("error"):
        return f"{payload['status']}: {payload['error']}"
    return payload["status"]


def create_executor(cfg_path, resolution):
    conf = read_yaml_config(cfg_path)
    if not check_required_env(conf, mode=resolution):
        raise RuntimeError("缺少必要环境变量，请先检查终端中的启动提示。")

    if resolution == "milvus":
        executor = MilvusExecutor(conf)
    else:
        executor = PipelineExecutor(conf)

    executor.build_query_engine()
    return executor


def initialize_app(cfg_path, resolution, state):
    next_state = dict(state or default_app_state())
    try:
        executor = create_executor(cfg_path, resolution)
    except Exception as exc:
        next_state.update(
            {
                "executor": None,
                "mode": resolution,
                "config_path": cfg_path,
                "initialized": False,
                "status": f"初始化失败: {exc}",
            }
        )
        return next_state, next_state["status"], next_state["status"]

    next_state.update(
        {
            "executor": executor,
            "mode": resolution,
            "config_path": cfg_path,
            "initialized": True,
            "status": "初始化完成",
        }
    )
    status = f"初始化完成，当前模式: {resolution}"
    return next_state, status, status


def run_user_query(state, question, show_debug):
    ready, message = ensure_ready_state(state)
    if not ready:
        return "", message, "暂无证据卡片。", ""

    executor = state["executor"]
    executor.set_debug(show_debug)
    payload = executor.query_ui(question)
    answer, status, sources_md, debug_md = map_query_payload_to_outputs(payload)
    if not show_debug:
        debug_md = ""
    return answer, status, sources_md, debug_md


def run_management_query(state, question):
    ready, message = ensure_ready_state(state)
    if not ready:
        return "", message, ""

    executor = state["executor"]
    executor.set_debug(True)
    payload = executor.query_ui(question)
    answer, status, _, debug_md = map_query_payload_to_outputs(payload)
    return answer, status, debug_md


def build_index_action(state, path, overwrite):
    ready, message = ensure_ready_state(state)
    if not ready:
        return message
    payload = state["executor"].build_index_ui(path, overwrite)
    return map_operation_payload_to_status(payload)


def delete_index_action(state, path):
    ready, message = ensure_ready_state(state)
    if not ready:
        return message
    payload = state["executor"].delete_file_ui(path)
    return map_operation_payload_to_status(payload)


def load_web_ui_doc():
    with open("docs/web_ui.md", "r", encoding="utf-8") as file_obj:
        return file_obj.read()


with gr.Blocks(title="History RAG Visual UI") as demo:
    app_state = gr.State(default_app_state())

    gr.Markdown("# History RAG 可视化前端")
    gr.Markdown("单页双标签界面：上手提问与知识库管理分离。")

    with gr.Tabs():
        with gr.Tab("问答助手"):
            gr.Markdown("## 问答助手")
            with gr.Row():
                qa_cfg_path = gr.Textbox(label="配置文件路径", value="cfgs/config.yaml")
                qa_mode = gr.Dropdown(choices=resolutions, label="模式", value="milvus")
            qa_init_btn = gr.Button("初始化")
            qa_status = gr.Textbox(label="状态", interactive=False, value="未初始化")
            qa_question = gr.Textbox(label="问题", lines=3, placeholder="请输入历史问题")
            qa_show_debug = gr.Checkbox(label="显示调试信息", value=False)
            qa_submit_btn = gr.Button("提交问题")
            qa_answer = gr.Markdown(label="回答")
            qa_sources = gr.Markdown(label="证据卡片", value="暂无证据卡片。")
            qa_debug = gr.Markdown(label="调试信息")

        with gr.Tab("知识库管理"):
            gr.Markdown("## 知识库管理")
            with gr.Row():
                admin_cfg_path = gr.Textbox(label="配置文件路径", value="cfgs/config.yaml")
                admin_mode = gr.Dropdown(choices=resolutions, label="模式", value="milvus")
            admin_init_btn = gr.Button("初始化")
            admin_status = gr.Textbox(label="状态", interactive=False, value="未初始化")

            gr.Markdown("### 构建索引")
            build_path = gr.Textbox(label="文件或目录路径")
            build_overwrite = gr.Checkbox(label="覆盖已有索引", value=False)
            build_btn = gr.Button("构建索引")
            build_status = gr.Textbox(label="构建结果", interactive=False)

            gr.Markdown("### 删除索引")
            delete_path = gr.Textbox(label="文件路径")
            delete_btn = gr.Button("删除索引")
            delete_status = gr.Textbox(label="删除结果", interactive=False)

            gr.Markdown("### 调试提问")
            admin_question = gr.Textbox(label="问题", lines=3)
            admin_query_btn = gr.Button("调试提问")
            admin_answer = gr.Markdown(label="回答")
            admin_query_status = gr.Textbox(label="查询状态", interactive=False)
            admin_debug = gr.Markdown(label="检索调试信息")

    gr.Markdown(load_web_ui_doc())

    qa_init_btn.click(
        initialize_app,
        inputs=[qa_cfg_path, qa_mode, app_state],
        outputs=[app_state, qa_status, admin_status],
    )
    admin_init_btn.click(
        initialize_app,
        inputs=[admin_cfg_path, admin_mode, app_state],
        outputs=[app_state, qa_status, admin_status],
    )
    qa_submit_btn.click(
        run_user_query,
        inputs=[app_state, qa_question, qa_show_debug],
        outputs=[qa_answer, qa_status, qa_sources, qa_debug],
    )
    build_btn.click(
        build_index_action,
        inputs=[app_state, build_path, build_overwrite],
        outputs=[build_status],
    )
    delete_btn.click(
        delete_index_action,
        inputs=[app_state, delete_path],
        outputs=[delete_status],
    )
    admin_query_btn.click(
        run_management_query,
        inputs=[app_state, admin_question],
        outputs=[admin_answer, admin_query_status, admin_debug],
    )


if __name__ == "__main__":
    demo.launch()
