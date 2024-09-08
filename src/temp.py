import ray
ray.init()

import image_gen.io.local_fs as fs
from utils import function_builder
from image_gen.constants import SENETENCE_ENCODER_MODEL_NAME
from sentence_transformers import SentenceTransformer

class SentenceEncoder_map_batches:
    def __init__(self):
        self.model = SentenceTransformer(SENETENCE_ENCODER_MODEL_NAME)

    def __call__(self, batch, text_key, result_name, keep_source=False):
        return batch


a = fs.read_metadata('../data/')
b = ray.data.from_pandas(a)
#seba = function_builder(SentenceEncoder_map_batches)
seba = SentenceEncoder_map_batches
print(
    b.map_batches(seba,concurrency=2, fn_kwargs={"text_key":"caption","result_name":"aaa"}).take(1),
)

