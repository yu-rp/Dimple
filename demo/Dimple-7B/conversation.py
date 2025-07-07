import os
import dataclasses
import base64
import copy
import hashlib
import datetime
from io import BytesIO
from PIL import Image
from typing import Any, List, Dict, Union
from dataclasses import field

from utils import LOGDIR


def pil2base64(img: Image.Image) -> str:
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def resize_img(img: Image.Image, max_len: int, min_len: int) -> Image.Image:
    max_hw, min_hw = max(img.size), min(img.size)
    aspect_ratio = max_hw / min_hw
    # max_len, min_len = 800, 400
    shortest_edge = int(min(max_len / aspect_ratio, min_len, min_hw))
    longest_edge = int(shortest_edge * aspect_ratio)
    W, H = img.size
    if H > W:
        H, W = longest_edge, shortest_edge
    else:
        H, W = shortest_edge, longest_edge
    return img.resize((W, H))


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

    roles: List[str] = field(
        default_factory=lambda: [
            Conversation.SYSTEM,
            Conversation.USER,
            Conversation.ASSISTANT,
        ]
    )
    mandatory_system_message = "Your name is Dimple. The first Discrete Diffusion Multimodal Large Language Model."
    system_message: str = "You are a helpful assistant."
    messages: List[Dict[str, Any]] = field(default_factory=lambda: [])
    max_image_limit: int = 2
    skip_next: bool = False
    streaming_placeholder: str = "•••"

    def get_system_message(self):
        if len(self.mandatory_system_message) == 0:
            return self.system_message
        else:
            return self.mandatory_system_message + "\n\n" + self.system_message

    def set_system_message(self, system_message: str):
        self.system_message = system_message
        return self
    
    def get_prompt(self):
        send_messages = [
            {
                "role": "system", 
                "content": self.get_system_message(),
            }
        ]
        for message in self.messages:
            if message["role"] == self.USER:
                user_message = {
                    "role": self.USER,
                    "content": message["content"],
                }
                if "image" in message:
                    user_message["image"] = []
                    for image in message["image"]:
                        user_message["image"].append(pil2base64(image))
                    
                    content = [{"type": "text", "text": message["content"]}]
                    for image_base64 in user_message["image"]:
                        content.append({
                            "type": "image", 
                            "image": f"data:image/jpeg;base64,{image_base64}"
                        })
                    send_messages.append({'role': self.USER, 'content': content})
                else:
                    send_messages.append(user_message)
            elif message["role"] == self.ASSISTANT:
                send_messages.append(
                    {"role": self.ASSISTANT, "content": message["content"]}
                )
            elif message["role"] == self.SYSTEM:
                send_messages.append(
                    {
                        "role": self.SYSTEM,
                        "content": message["content"],
                    }
                )
            else:
                raise ValueError(f"Invalid role: {message['role']}")
        return send_messages

    def append_message(
        self,
        role,
        content,
        image_list=None,
    ):
        self.messages.append(
            {
                "role": role,
                "content": content,
                "image": [] if image_list is None else image_list,
                # "filenames": save_filenames,
            }
        )

    def get_images(
        self,
        return_copy=False,
        return_base64=False,
        source: Union[str, None] = None,
    ):
        assert source in [self.USER, self.ASSISTANT, None], f"Invalid source: {soure}"
        images = []
        for i, msg in enumerate(self.messages):
            if source and msg["role"] != source:
                continue

            for image in msg.get("image", []):
                # org_image = [i.copy() for i in image]
                if return_copy:
                    image = image.copy()

                if return_base64:
                    image = pil2base64(image)

                images.append(image)

        return images

    def to_gradio_chatbot(self):
        ret = []
        for i, msg in enumerate(self.messages):
            if msg["role"] == self.SYSTEM:
                continue

            alt_str = (
                "user upload image" if msg["role"] == self.USER else "output image"
            )
            image = msg.get("image", [])
            if not isinstance(image, list):
                images = [image]
            else:
                images = image

            img_str_list = []
            for i in range(len(images)):
                image = resize_img(
                    images[i],
                    400,
                    200,
                )
                img_b64_str = pil2base64(image)
                W, H = image.size
                img_str = f'<img src="data:image/png;base64,{img_b64_str}" alt="{alt_str}" style="width: {W}px; max-width:none; max-height:none"></img>'
                # img_str = (
                #     f'<img src="data:image/png;base64,{img_b64_str}" alt="{alt_str}" />'
                # )
                img_str_list.append(img_str)
            
            if ('\[' in msg["content"] and '\]' in msg["content"]) or ('\(' in msg["content"] and '\)' in msg["content"]):
                # 行内公式 or 行间公式
                content = msg["content"].replace('\[', '$$').replace('\]', '$$').replace('\(', '$$').replace('\)', '$$')
                content = content.split('$$')
                for i in range(len(content)):
                    if i % 2:
                        content[i] = content[i].strip()
                content = '$$'.join(content)
                print('content:', content)
                # content = (
                #     r"<span>" + content + r"</span>"
                #     r"<script type='text/javascript'>"
                #     r"MathJax.typesetPromise();"
                #     r"</script>"
                # )
            else:
                content = msg["content"]
            if msg["role"] == self.USER:
                msg_str = " ".join(img_str_list) + content
                ret.append([msg_str, None])
            else:
                msg_str = content + " ".join(img_str_list)
                ret[-1][-1] = msg_str
        return ret

    def update_message(self, role, content, image=None, idx=-1):
        assert len(self.messages) > 0, "No message in the conversation."

        idx = (idx + len(self.messages)) % len(self.messages)

        assert (
            self.messages[idx]["role"] == role
        ), f"Role mismatch: {role} vs {self.messages[idx]['role']}"

        self.messages[idx]["content"] = content
        if image is not None:
            if image not in self.messages[idx]["image"]:
                self.messages[idx]["image"] = []
            if not isinstance(image, list):
                image = [image]
            self.messages[idx]["image"].extend(image)

    def return_last_message(self):
        return self.messages[-1]["content"]

    def end_of_current_turn(self):
        assert len(self.messages) > 0, "No message in the conversation."
        assert (
            self.messages[-1]["role"] == self.ASSISTANT
        ), f"It should end with the message from assistant instead of {self.messages[-1]['role']}."

        if self.messages[-1]["content"][-1] != self.streaming_placeholder:
            return

        self.update_message(self.ASSISTANT, self.messages[-1]["content"][:-1], None)

    def copy(self):
        return Conversation(
            mandatory_system_message=self.mandatory_system_message,
            system_message=self.system_message,
            roles=copy.deepcopy(self.roles),
            messages=copy.deepcopy(self.messages),
        )

    def dict(self):
        """
        all_images = state.get_images()
        all_image_hash = [hashlib.md5(image.tobytes()).hexdigest() for image in all_images]
        t = datetime.datetime.now()
        for image, hash in zip(all_images, all_image_hash):
            filename = os.path.join(
                LOGDIR, "serve_images", f"{t.year}-{t.month:02d}-{t.day:02d}", f"{hash}.jpg"
            )
            if not os.path.isfile(filename):
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                image.save(filename)
        """
        messages = []
        for message in self.messages:
            images = []
            for image in message.get("image", []):
                filename = self.save_image(image)
                images.append(filename)

            messages.append(
                {
                    "role": message["role"],
                    "content": message["content"],
                    "image": images,
                }
            )
            if len(images) == 0:
                messages[-1].pop("image")

        return {
            "mandatory_system_message": self.mandatory_system_message,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": messages,
        }

    def save_image(self, image: Image.Image) -> str:
        t = datetime.datetime.now()
        image_hash = hashlib.md5(image.tobytes()).hexdigest()
        filename = os.path.join(
            LOGDIR,
            "serve_images",
            f"{t.year}-{t.month:02d}-{t.day:02d}",
            f"{image_hash}.jpg",
        )
        if not os.path.isfile(filename):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            image.save(filename)

        return filename



