from sentence_transformers import SentenceTransformer

class SentenceEncoder_map_batches:
    def __init__(self, SENETENCE_ENCODER_MODEL_NAME):
        self.model = SentenceTransformer(SENETENCE_ENCODER_MODEL_NAME)

    def __call__(self, batch, text_key, result_name, keep_source=False):
        sentences = batch.pop(text_key)
        batch[result_name] = self.model.encode(sentences)

        if keep_source:
            batch[text_key] = sentences

        return batch
