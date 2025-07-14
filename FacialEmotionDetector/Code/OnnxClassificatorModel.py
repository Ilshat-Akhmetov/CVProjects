import numpy as np
import onnxruntime as ort
from typing import List, Tuple


class OnnxModel:
    def __init__(self, filename: str):
        self.ort_session = ort.InferenceSession(filename,
                                               providers=['CPUExecutionProvider'])
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.n_classes = len(self.classes)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        input: np.ndarray with shape (batch_size, 1, img_size, img_size)
        output: np.ndarray with shape (batch_size, n_classes)
        """
        input_name = self.ort_session.get_inputs()[0].name
        output_name = self.ort_session.get_outputs()[0].name
        outputs = self.ort_session.run([output_name], {input_name: input_data})
        result = outputs[0]
        return result

    def get_probs(self, input_data: np.ndarray) -> np.ndarray:
        """
        input: np.ndarray with shape (batch_size, n_classes)
        output: np.ndarray with shape (batch_size, n_classes)
        """
        assert len(input_data.shape) == 2, 'input must be 2dim'
        max_v = np.max(input_data, axis=1, keepdims=True)
        normed_data = input_data - max_v
        data_exp = np.exp(normed_data)
        sum_per_row = np.sum(data_exp, axis=1, keepdims=True)
        return data_exp / sum_per_row

    def get_img_classes_probs(self, input_data: np.ndarray) -> Tuple[List[float], List[str]]:
        """
        input: np.ndarray with shape (1, 1, img_size, img_size)
        output: Tuple: List with probs, List with most likely classes
        """
        assert input_data.shape[0]==1, 'methods accepts only one image'
        max_outp_classes = 3
        nn_outp = self.forward(input_data)
        probs = self.get_probs(nn_outp)[0]
        probs_inds = list(zip(range(self.n_classes), probs))
        probs_inds = sorted(probs_inds, key=lambda x: -x[1])[:max_outp_classes]
        sorted_probs = [x[1].item() for x in probs_inds]
        sorted_cls = [self.classes[x[0]] for x in probs_inds]
        return sorted_probs, sorted_cls


            
            
            