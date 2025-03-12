import torch
from vllm.worker.worker import Worker

from replaybuf import IS_NPU_AVAILABLE
from openrlhf.utils.distributed_util import init_process_group
from openrlhf.utils.logging_utils import init_logger
from openrlhf.accelerator import current_accelerator


if IS_NPU_AVAILABLE:
    from vllm_ascend.worker.worker import NPUWorker as Worker

logger = init_logger(__name__)


class WorkerWrap(Worker):
    pass
