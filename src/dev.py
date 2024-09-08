import ray
ray.init()
import matplotlib.pyplot as plt
import image_gen.io.local_fs as lf
import image_gen.preprocessing.sentence_embedding as se
import image_gen.preprocessing.image_rescaling as ir
a = lf.read_metadata('../data')
b = ray.data.from_pandas(a)
c = b.map_batches(se.SentenceEncoder_map_batches, concurrency=2, fn_kwargs={'text_key':'caption','result_name':'emb'})

d = b.map_batches(ir.load_and_resize_map_batches, fn_kwargs={'size':(128,128),'coco_path':'/home/przemek/Desktop/image-gen/data', 'filename_key':'file_name', 'result_name':'image'})
e = d.take_batch()
