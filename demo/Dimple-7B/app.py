try:
    import spaces
except ImportError:
    # Local run: define dummy decorator
    class spaces:
        @staticmethod
        def GPU(duration=10):
            def dummy(func):
                return func
            return dummy

import argparse
import json
import time

import gradio as gr
from filelock import FileLock
from PIL import Image
import threading

from utils import (
    build_logger,
    server_error_msg,
    violates_moderation,
    moderation_msg,
    get_log_filename,
)
from conversation import Conversation
from model import (
    FullSequenceStreamer,
    get_model,
)

# seed for reproducibility
import random
import numpy as np
import torch
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

logger = build_logger("dimple", "dimple.log")

no_change_btn = gr.Button()
enable_btn = gr.Button(interactive=True)
disable_btn = gr.Button(interactive=False)


@spaces.GPU(duration=10)
def make_zerogpu_happy():
    pass


def write2file(path, content):
    lock = FileLock(f"{path}.lock")
    with lock:
        with open(path, "a") as fout:
            fout.write(content)

model, processor = get_model("cuda:0")

get_window_url_params = """
function() {
    const params = new URLSearchParams(window.location.search);
    url_params = Object.fromEntries(params);
    console.log(url_params);
    return url_params;
    }
"""


def init_state(state=None):
    if state is not None:
        del state
    return Conversation()

def vote_last_response(state, liked, request: gr.Request):
    conv_data = {
        "tstamp": round(time.time(), 4),
        "like": liked,
        "model": '"rp-yu/Dimple-7B"',
        "state": state.dict(),
        "ip": request.client.host,
    }
    write2file(get_log_filename(), json.dumps(conv_data) + "\n")


def upvote_last_response(state, request: gr.Request):
    logger.info(f"upvote. ip: {request.client.host}")
    vote_last_response(state, True, request)
    textbox = gr.MultimodalTextbox(value=None, interactive=True)
    return (textbox,) + (disable_btn,) * 3


def downvote_last_response(state, request: gr.Request):
    logger.info(f"downvote. ip: {request.client.host}")
    vote_last_response(state, False, request)
    textbox = gr.MultimodalTextbox(value=None, interactive=True)
    return (textbox,) + (disable_btn,) * 3


def vote_selected_response(
    state, request: gr.Request, data: gr.LikeData
):
    logger.info(
        f"Vote: {data.liked}, index: {data.index}, value: {data.value} , ip: {request.client.host}"
    )
    conv_data = {
        "tstamp": round(time.time(), 4),
        "like": data.liked,
        "index": data.index,
        "model": 'rp-yu/Dimple-7B',
        "state": state.dict(),
        "ip": request.client.host,
    }
    write2file(get_log_filename(), json.dumps(conv_data) + "\n")
    return


def flag_last_response(state, request: gr.Request):
    logger.info(f"flag. ip: {request.client.host}")
    vote_last_response(state, "flag", request)
    textbox = gr.MultimodalTextbox(value=None, interactive=True)
    return (textbox,) + (disable_btn,) * 3


def regenerate(state, image_process_mode, request: gr.Request):
    logger.info(f"regenerate. ip: {request.client.host}")
    # state.messages[-1][-1] = None
    state.update_message(Conversation.ASSISTANT, content='', image=None, idx=-1)
    prev_human_msg = state.messages[-2]
    if type(prev_human_msg[1]) in (tuple, list):
        prev_human_msg[1] = (*prev_human_msg[1][:2], image_process_mode)
    state.skip_next = False
    textbox = gr.MultimodalTextbox(value=None, interactive=True)
    return (state, state.to_gradio_chatbot(), textbox) + (disable_btn,) * 5


def clear_history(request: gr.Request):
    logger.info(f"clear_history. ip: {request.client.host}")
    state = init_state()
    textbox = gr.MultimodalTextbox(value=None, interactive=True)
    return (state, state.to_gradio_chatbot(), textbox) + (disable_btn,) * 5


def add_text(state, message, system_prompt, request: gr.Request):
    print(f"state: {state}")
    if not state:
        state = init_state()
    images = message.get("files", [])
    text = message.get("text", "").strip()
    logger.info(f"add_text. ip: {request.client.host}. len: {len(text)}")
    # import pdb; pdb.set_trace()
    textbox = gr.MultimodalTextbox(value=None, interactive=False)
    if len(text) <= 0 and len(images) == 0:
        state.skip_next = True
        return (state, state.to_gradio_chatbot(), textbox) + (no_change_btn,) * 5
    if args.moderate:
        flagged = violates_moderation(text)
        if flagged:
            state.skip_next = True
            textbox = gr.MultimodalTextbox(
                value={"text": moderation_msg}, interactive=True
            )
            return (state, state.to_gradio_chatbot(), textbox) + (no_change_btn,) * 5
    images = [Image.open(path).convert("RGB") for path in images]

    if len(images) > 0 and len(state.get_images(source=state.USER)) > 0:
        state = init_state(state)
    state.set_system_message(system_prompt)
    state.append_message(Conversation.USER, text, images)
    state.skip_next = False
    return (state, state.to_gradio_chatbot(), textbox) + (
        disable_btn,
    ) * 5


def http_bot(
    state,
    temperature,
    top_p,
    p_threshold,
    alg_temp,
    max_new_tokens,
    steps,
    alg,
):
    start_tstamp = time.time()
    if hasattr(state, "skip_next") and state.skip_next:
        # This generate call is skipped due to invalid inputs
        yield (
            state,
            state.to_gradio_chatbot(),
            gr.MultimodalTextbox(interactive=False),
        ) + (no_change_btn,) * 5
        return

    all_images = state.get_images(source=state.USER)
    all_image_paths = [state.save_image(image) for image in all_images]
    
    if len(all_images) == 0:
        all_images = None

    messages = state.get_prompt()
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, add_vision_id=False
    )

    inputs = processor(
            text=text,
            images=all_images,
            videos=None,
            padding="longest",
            return_tensors="pt",
        ).to(model.device)
    input_ids = inputs.pop("input_ids")

    streamer = FullSequenceStreamer(
        processor.tokenizer,
        timeout=10,
        skip_special_tokens=True,
    )

    def run_generate():
        output = model.diffusion_generate(
            input_ids,
            max_new_tokens=int(max_new_tokens),
            output_history=True,
            return_dict_in_generate=True,
            steps=int(steps),
            temperature=float(temperature),
            top_p=float(top_p),
            alg=alg,
            alg_temp = float(alg_temp),
            use_cache=True,
            alg_p_threshold=float(p_threshold),
            use_original_confidence=True,
            decoding_pipeline="dim",
            streamer = streamer,
            **inputs
        )
    
    thread = threading.Thread(target=run_generate)
    logger.info(f"==== wait for first token ====\n")
    state.append_message(Conversation.ASSISTANT, state.streaming_placeholder)
    yield (
        state,
        state.to_gradio_chatbot(),
        gr.MultimodalTextbox(interactive=False),
    ) + (disable_btn,) * 5

    num_steps = 0
    start_tstamp = time.time()
    thread.start()

    try:
        # Stream output
        for ans in streamer:
            if len(ans) > 1:
                ans = "\n".join(ans)
            else:
                ans = ans[0]
                
            state.update_message(Conversation.ASSISTANT, ans, None)
            num_steps += 1
            yield (
                state,
                state.to_gradio_chatbot(),
                gr.MultimodalTextbox(interactive=False),
            ) + (disable_btn,) * 5
        end_tstamp = time.time()
        total_time = end_tstamp - start_tstamp
        tps = int(max_new_tokens) / total_time
        stat_info = f"\n\n[#Tokens: {int(max_new_tokens)}, #Steps: {int(num_steps)}, TPS: {tps:.2f} tokens/s, Latency: {total_time:.2f}s]"
        state.update_message(Conversation.ASSISTANT, ans+stat_info, None)
        yield (
                state,
                state.to_gradio_chatbot(),
                gr.MultimodalTextbox(interactive=False),
            ) + (disable_btn,) * 5
    except Exception as e:
        state.update_message(Conversation.ASSISTANT, server_error_msg, None)
        yield (
            state,
            state.to_gradio_chatbot(),
            gr.MultimodalTextbox(interactive=True),
        ) + (
            disable_btn,
            disable_btn,
            disable_btn,
            enable_btn,
            enable_btn,
        )
        return

    state.end_of_current_turn()

    yield (
        state,
        state.to_gradio_chatbot(),
        gr.MultimodalTextbox(interactive=True),
    ) + (enable_btn,) * 5

    finish_tstamp = time.time()
    logger.info(f"{ans}")
    data = {
        "tstamp": round(finish_tstamp, 4),
        "like": None,
        "model": "rp-yu/Dimple-7B",
        "start": round(start_tstamp, 4),
        "finish": round(start_tstamp, 4),
        "state": state.dict(),
        "images": all_image_paths,
    }
    write2file(get_log_filename(), json.dumps(data) + "\n")


title_html = """
<div style="width:100%; max-width:600px; margin:auto;">
<img src="https://cdn-uploads.huggingface.co/production/uploads/635364b3c41f548fe39db945/Iny16670lQgUwURiUfP-i.png" style="width:100%;"><br>
<a href="https://arxiv.org/abs/">[📜 Dimple Paper]</a><br>
<a href="https://github.com/yu-rp/Dimple">[🌟 Github]</a><br>  
<a href="https://huggingface.co/rp-yu/Dimple-7B">[🤗 Huggingface Model]</a><br>
<a href="https://huggingface.co/spaces/rp-yu/dimple">[💬 Huggingface Demo]</a><br>
</div>
"""


tos_markdown = """
<p align="right">
Acknowledgement: This demo is built upon the Hugging Face Space of <a href="https://huggingface.co/spaces/OpenGVLab/InternVL" target="_blank">InternVL</a>.
</p>
"""


# .gradio-container {margin: 5px 10px 0 10px !important};
block_css = """
.gradio-container {margin: 0.1% 1% 0 1% !important; max-width: 98% !important;};
#buttons button {
    min-width: min(120px,100%);
}

.gradient-text {
    font-size: 28px;
    width: auto;
    font-weight: bold;
    background: linear-gradient(45deg, red, orange, yellow, green, blue, indigo, violet);
    background-clip: text;
    -webkit-background-clip: text;
    color: transparent;
}

.plain-text {
    font-size: 22px;
    width: auto;
    font-weight: bold;
}
"""


def build_demo():
    textbox = gr.MultimodalTextbox(
        interactive=True,
        file_types=["image"],
        placeholder="Enter message or upload file...",
        show_label=False,
    )

    with gr.Blocks(
        title="Dimple-7B",
        theme=gr.themes.Default(),
        css=block_css,
    ) as demo:
        state = gr.State()

        with gr.Row():
            with gr.Column(scale=2):
                gr.HTML(title_html)

                with gr.Accordion("Settings", open=False) as setting_row:
                    system_prompt = gr.Textbox(
                        value="You are a helpful assistant.",
                        label="System Prompt",
                        interactive=True,
                    )
                    temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.0,
                        step=0.1,
                        interactive=True,
                        label="Temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.0,
                        step=0.1,
                        interactive=True,
                        label="Top P",
                    )
                    alg = gr.Radio(
                        choices=["origin", "maskgit_plus", "entropy"],
                        value="origin",
                        label="Selection Algorithm",
                        interactive=True,
                    )
                    p_threshold = gr.Slider(
                        minimum=0.,
                        maximum=1.0,
                        value=0.95,
                        step=0.01,
                        interactive=True,
                        label="Probability threshold for Confident Decoding",
                    )
                    alg_temp = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=0.4,
                        step=0.1,
                        interactive=True,
                        label="Temperature for Selectiion Algorithm",
                    )
                    max_new_tokens = gr.Slider(
                        minimum=1,
                        maximum=128,
                        value=64,
                        step=2,
                        interactive=True,
                        label="Max output tokens",
                    )
                    steps = gr.Slider(
                        minimum=1,
                        maximum=128,
                        value=64,
                        step=2,
                        interactive=True,
                        label="Number of decoding steps",
                    )

                examples = gr.Examples(
                    examples=[
                        [
                            {
                                "files": [
                                    "gallery/1_resize.jpg",
                                ],
                                "text": "Please help me describe the image.",
                            }
                        ],
                        [
                            {
                                "files": [
                                    "gallery/2_resize.jpg",
                                ],
                                "text": "First please perform reasoning, and think step by step to provide best answer to the following question: Is this airplane taking off or landing?",
                            }
                        ],
                        [
                            {
                                "files": [
                                    "gallery/3_resize.jpg",
                                ],
                                "text": "First please perform reasoning, and think step by step to provide best answer to the following question: What is the lamp on, a side table or a nightstand?",
                            }
                        ],
                    ],
                    inputs=[textbox],
                )

            with gr.Column(scale=8):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    label="Dimple-7B",
                    height=580,
                    show_copy_button=True,
                    show_share_button=True,
                    avatar_images=[
                        "assets/human.png",
                        "assets/assistant.png",
                    ],
                    bubble_full_width=False,
                )
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(value="Send", variant="primary")
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    flag_btn = gr.Button(value="⚠️  Flag", interactive=False)
                    # stop_btn = gr.Button(value="⏹️  Stop Generation", interactive=False)
                    regenerate_btn = gr.Button(
                        value="🔄  Regenerate", interactive=False
                    )
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)

        gr.Markdown(tos_markdown)
        url_params = gr.JSON(visible=False)

        # Register listeners
        btn_list = [upvote_btn, downvote_btn, flag_btn, regenerate_btn, clear_btn]
        upvote_btn.click(
            upvote_last_response,
            [state],
            [textbox, upvote_btn, downvote_btn, flag_btn],
        )
        downvote_btn.click(
            downvote_last_response,
            [state],
            [textbox, upvote_btn, downvote_btn, flag_btn],
        )
        chatbot.like(
            vote_selected_response,
            [state],
            [],
        )
        flag_btn.click(
            flag_last_response,
            [state],
            [textbox, upvote_btn, downvote_btn, flag_btn],
        )
        regenerate_btn.click(
            regenerate,
            [state, system_prompt],
            [state, chatbot, textbox] + btn_list,
        ).then(
            http_bot,
            [
                state,
                temperature,
                top_p,
                p_threshold,
                alg_temp,
                max_new_tokens,
                steps,
                alg,
            ],
            [state, chatbot, textbox] + btn_list,
        )
        clear_btn.click(clear_history, None, [state, chatbot, textbox] + btn_list)

        textbox.submit(
            add_text,
            [state, textbox, system_prompt],
            [state, chatbot, textbox] + btn_list,
        ).then(
            http_bot,
            [
                state,
                temperature,
                top_p,
                p_threshold,
                alg_temp,
                max_new_tokens,
                steps,
                alg,
            ],
            [state, chatbot, textbox] + btn_list,
        )
        submit_btn.click(
            add_text,
            [state, textbox, system_prompt],
            [state, chatbot, textbox] + btn_list,
        ).then(
            http_bot,
            [
                state,
                temperature,
                top_p,
                p_threshold,
                alg_temp,
                max_new_tokens,
                steps,
                alg,
            ],
            [state, chatbot, textbox] + btn_list,
        )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--concurrency-count", type=int, default=4)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--moderate", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    logger.info(args)
    demo = build_demo()
    demo.queue(api_open=False).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        max_threads=args.concurrency_count,
    )
