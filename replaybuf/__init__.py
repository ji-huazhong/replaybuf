import import_export


def is_npu_available():
    return importlib.util.find_spec("torch_npu")

ACCELERATOR_TYPE = "GPU"

if IS_NPU_AVAILABLE:
    ACCELERATOR_TYPE = "NPU"
