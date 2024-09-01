from sentence_transformers import SentenceTransformer
from image_gen.constants import SENETENCE_ENCODER_MODEL_NAME

class SentenceEncoder_map_batches:
    def __init__(self):
        self.model = SentenceTransformer(SENETENCE_ENCODER_MODEL_NAME)

    def __call__(self, batch, text_key, result_name, keep_source=False):
        sentences = batch.pop(text_key)
        batch[result_name] = self.model.encode(sentences)

        if keep_source:
            batch[text_key] = sentences

        return batch
