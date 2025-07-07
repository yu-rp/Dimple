
from typing import Any, Optional, Union
from dataclasses import asdict, dataclass, field, fields

@dataclass
class DLMTrainingArguments:
    mask_token_id: int = field(
        default=None,
        metadata={
            "help": "ID for the special mask token"
        },
    )
    transition_mode: Optional[str] = field(
        default='linear',
        metadata={
            "help": "Transition mode for the mask token, support only linear now"
        },
    )
    seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "Sequence length for the model"
        },
    )
    loss_rescale: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Rescale the loss for the model"
        },
    )
    mask_strategy: Optional[str] = field(
        default="single",
        metadata={
            "help": "Mask strategy for the model, support only single and dual now"
        },
    )
    shift: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether the MLLM shift the labels"
        },
    )
    order_token: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether the MLLM use order token"
        },
    )
    order_token_weight_function: Optional[str] = field(
        default="linear",
        metadata={
            "help": "Order token weight for the model"
        },
    )
    linear_clamp_shift: Optional[float] = field(
        default=0.8,
        metadata={
            "help": "Clamp shift used in the linear order token weight function"
        },
    )
    linear_clamp_max: Optional[float] = field(
        default=2.0,
        metadata={
            "help": "Clamp max used in the linear order token weight function"
        },
    )
    linear_clamp_min: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "Clamp min used in the linear order token weight function"
        },
    )
