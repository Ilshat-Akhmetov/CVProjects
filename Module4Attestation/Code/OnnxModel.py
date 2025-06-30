import numpy as np
import onnxruntime as ort


class OnnxModel:
    def __init__(self, filename: str):
        self.ort_session = ort.InferenceSession(filename,
                                               providers=['CPUExecutionProvider'])

    def __call__(self, input_data: np.ndarray) -> np.ndarray:
        """
        input: np.ndarray with shape (batch_size, 1, 96, 96)
        output: np.ndarray with shape (batch_size, 15, 2)
        """
        input_name = self.ort_session.get_inputs()[0].name
        output_name = self.ort_session.get_outputs()[0].name
        outputs = self.ort_session.run([output_name], {input_name: input_data})
        result = outputs[0]
        return result