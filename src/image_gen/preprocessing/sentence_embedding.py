from sentence_transformers import SentenceTransformer
import torch


class SentenceEncoder_map_batches:
    def __init__(self, SENETENCE_ENCODER_MODEL_NAME):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(SENETENCE_ENCODER_MODEL_NAME).to(self.device)
        print("sentence encoder initialized")

    def __call__(self, batch, text_key, result_name, keep_source=False):
        sentences = batch.pop(text_key)
        batch[result_name] = self.model.encode(sentences)

        if keep_source:
            batch[text_key] = sentences

        return batch
