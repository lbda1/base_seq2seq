__author__ = 'liyang54'
#coding=utf-8
import tensorflow as tf
from data_process import extract_character_vocab
batch_size = 128
# batch_size = FLAGS.batch_size

with open('letters_source.txt', 'r', encoding='utf-8') as f:
    source_data = f.read()

with open('letters_target.txt', 'r', encoding='utf-8') as f:
    target_data = f.read()

source_int_to_letter, source_letter_to_int = extract_character_vocab(source_data)
target_int_to_letter, target_letter_to_int = extract_character_vocab(target_data)

def source_to_seq(text):
    '''
    对源数据进行转换
    '''
    sequence_length = 7
    return [source_letter_to_int.get(word, source_letter_to_int['<UNK>']) for word in text] + [source_letter_to_int['<PAD>']]*(sequence_length-len(text))


input_word = 'coamm'
text = source_to_seq(input_word)
print(text)
# [15, 16, 9, 9, 16, 10, 0]
# [23, 18, 16, 16, 18, 17, 0]
# [17, 29, 5, 5, 29, 6, 0]
checkpoint = "model_checkpoint/trained_model.ckpt"

# loaded_graph = tf.Graph()
with tf.Session() as sess:
# with tf.Session(graph=loaded_graph) as sess:
    # 加载模型
    saver = tf.train.import_meta_graph(checkpoint + '.meta')
    saver.restore(sess, checkpoint)
    graph = tf.get_default_graph()

    input_data = graph.get_tensor_by_name('inputs:0')
    logits = graph.get_tensor_by_name('predictions:0')
    source_sequence_length = graph.get_tensor_by_name('source_sequence_length:0')
    target_sequence_length = graph.get_tensor_by_name('target_sequence_length:0')
    feed_dic = {input_data: [text]*batch_size,
                target_sequence_length: [len(input_word)]*batch_size,
                source_sequence_length: [len(input_word)]*batch_size}
    answer_logits = sess.run(logits, feed_dic)[0]


pad = source_letter_to_int["<PAD>"]

print('原始输入:', input_word)

print('\nSource')
print('  Word 编号:    {}'.format([i for i in text]))
print('  Input Words: {}'.format(" ".join([source_int_to_letter[i] for i in text])))

print('\nTarget')
print('  Word 编号:       {}'.format([i for i in answer_logits if i != pad]))
print('  Response Words: {}'.format(" ".join([target_int_to_letter[i] for i in answer_logits if i != pad])))