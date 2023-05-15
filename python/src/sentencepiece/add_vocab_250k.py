import sentencepiece_model_pb2 as model
# import os
# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
m = model.ModelProto()
m.ParseFromString(open("250k.model", "rb").read())
# import pudb; pu.db
print(m.trainer_spec)
# print(m.normalizer_spec)

# m.ParseFromString(open("tokenizer.model", "rb").read())
# print(m)

# fix spm model
m.normalizer_spec.remove_extra_whitespaces = False
m.trainer_spec.split_digits = True
# m.trainer_spec.byte_fallback = True


del_pieces = []
for idx, tok in enumerate(m.pieces):
    if any(char.isnumeric() for char in tok.piece):
        # continue
        del_pieces.append(idx)
    else:
        continue
for idx in del_pieces[::-1]:
    del m.pieces[idx]


special_tokens = [f'{i}' for i in range(0, 10)]
special_tokens += ['▁' * i for i in range(16, 1, -1)]

# special_tokens = ['▁' * i for i in range(20, 21)]
print(special_tokens)

idx = 0
for token in special_tokens:
    new_token = model.ModelProto().SentencePiece()
    new_token.piece = token
    new_token.score = -1
    idx += 1
    m.pieces.append(new_token)


with open('250k.fix.model', 'wb') as f:
    f.write(m.SerializeToString())

import sentencepiece as spm

tokenizer = spm.SentencePieceProcessor(model_file=r"tokenizer.model")

# vocab_size = tokenizer.get_piece_size()
# vocab = [tokenizer.id_to_piece(i) for i in range(vocab_size)]
# for i in vocab:
#     print(i)
sentence = 'an \t image of                       a person 2342354345324 23432.5453 -32234 -3423.343 \n\n-123e4 我爱北京天安门'
ids = tokenizer.encode(sentence, out_type=str)
print(ids)
print(len('▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁'))

our_tokenizer = spm.SentencePieceProcessor(model_file=r"250k.fix.model")
ids = our_tokenizer.encode(sentence, out_type=str)
print(ids)
