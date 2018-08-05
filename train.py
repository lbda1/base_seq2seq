#coding=utf-8
import os
from data_process import extract_character_vocab
# https://blog.csdn.net/leadai/article/details/78809788
# https://www.jianshu.com/p/83443b2baf27
# https://zhuanlan.zhihu.com/p/27608348
__author__ = 'liyang54'
import numpy as np
from seq2seq_model import seq2seq_model
import tensorflow as tf
# 超参数
# Number of Epochs
epochs = 60
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 50
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 15
decoding_embedding_size = 15
# Learning Rate
learning_rate = 0.001
tf.flags.DEFINE_integer("num_checkpoints", 30, "Number of checkpoints to store ")
tf.flags.DEFINE_integer("batch_size", 128, "Batch Size ")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.flags.DEFINE_boolean("debug", False, "Run with tf debugger")

FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()

with open('letters_source.txt', 'r', encoding='utf-8') as f:
    source_data = f.read()

with open('letters_target.txt', 'r', encoding='utf-8') as f:
    target_data = f.read()

source_int_to_letter, source_letter_to_int = extract_character_vocab(source_data)
target_int_to_letter, target_letter_to_int = extract_character_vocab(target_data)


# 对字母进行转换
source_int = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>'])
               for letter in line] for line in source_data.split('\n')]
target_int = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>'])
               for letter in line] + [target_letter_to_int['<EOS>']] for line in target_data.split('\n')]

# 将数据集分割为train和validation
train_source = source_int[batch_size:]
train_target = target_int[batch_size:]
# 留出一个batch进行验证
valid_source = source_int[:batch_size]
valid_target = target_int[:batch_size]

display_step = 50 # 每隔50轮输出loss

#模型参数定义和占位
def get_inputs():
    '''
    模型输入tensor
    '''
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length

def pad_sentence_batch(sentence_batch, pad_int):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    '''
    定义生成器，用来获取batch
    '''
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        # 补全序列
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # 记录每条记录的长度
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths

# 获得模型输入
input_data, targets, lr, target_sequence_length, max_target_sequence_length, source_sequence_length = get_inputs()

# 上面都是数据预处理
# 构建图
#有默认的graph
# with tf.Graph().as_default():
session_conf = tf.ConfigProto(
  allow_soft_placement=FLAGS.allow_soft_placement,
  log_device_placement=FLAGS.log_device_placement
)
session_conf.gpu_options.allow_growth = True


sess = tf.Session(config=session_conf)
with sess.as_default():
    training_decoder_output, predicting_decoder_output = seq2seq_model(input_data,
                                                                      targets,
                                                                      lr,
                                                                      target_sequence_length,
                                                                      max_target_sequence_length,
                                                                      source_sequence_length,
                                                                      len(source_letter_to_int),
                                                                      len(target_letter_to_int),
                                                                      encoding_embedding_size,
                                                                      decoding_embedding_size,
                                                                      rnn_size,
                                                                      num_layers,
                                                                      target_letter_to_int,
                                                                      batch_size,
                                                                      decoding_embedding_size)

    # tf.identity是返回一个一模一样新的tensor的op，这会增加一个新节点到gragh中
    # tf.contrib.seq2seq.dynamic_decode
    # 用于构造一个动态的decoder，返回的内容是：
    # (final_outputs, final_state, final_sequence_lengths).
    # 其中，final_outputs是一个namedtuple，里面包含两项(rnn_outputs, sample_id)
    # rnn_output: [batch_size, decoder_targets_length, vocab_size]，保存decode每个时刻每个单词的概率，可以用来计算loss
    # sample_id: [batch_size], tf.int32，保存最终的编码结果。可以表示最后的答案
    # 作者：石晓文的学习日记
    training_logits = tf.identity(training_decoder_output.rnn_output, 'logits')
    predicting_logits = tf.identity(predicting_decoder_output.sample_id, name='predictions')

    masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')

    with tf.name_scope("optimization"):

        # Loss function
        cost = tf.contrib.seq2seq.sequence_loss(
            training_logits,
            targets,
            masks)

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)





    # 用于验证的batch
    (valid_targets_batch, valid_sources_batch, valid_targets_lengths, valid_sources_lengths) = next(get_batches(valid_target, valid_source, batch_size,
                               source_letter_to_int['<PAD>'],
                               target_letter_to_int['<PAD>']))



    #训练部分
    checkpoint = "model_checkpoint/trained_model.ckpt"
    sess.run(tf.global_variables_initializer())

    #checkpoint 保存
    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    # out_dir = "runs"
    # checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    # checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir)
    # saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
    #
    # # Write vocabulary
    # vocab_processor.save(os.path.join(out_dir, "vocab"))

    for epoch_i in range(1, epochs+1):
        for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                get_batches(train_target, train_source, batch_size,
                           source_letter_to_int['<PAD>'],
                           target_letter_to_int['<PAD>'])):

            feed_dic = {input_data: sources_batch,
                 targets: targets_batch,
                 lr: learning_rate,
                 target_sequence_length: targets_lengths,
                 source_sequence_length: sources_lengths}

            _, loss = sess.run([train_op, cost], feed_dic)


            if batch_i % display_step == 0:

                # 计算validation loss
                feed_dic = {input_data: valid_sources_batch,
                 targets: valid_targets_batch,
                 lr: learning_rate,
                 target_sequence_length: valid_targets_lengths,
                 source_sequence_length: valid_sources_lengths}

                validation_loss = sess.run([cost], feed_dic)

                print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                      .format(epoch_i,
                              epochs,
                              batch_i,
                              len(train_source) // batch_size,
                              loss,
                              validation_loss[0]))
    # 保存模型
    saver = tf.train.Saver()
    saver.save(sess, checkpoint)
    print('Model Trained and Saved')
