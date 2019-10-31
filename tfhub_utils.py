import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece



class TFHubContext:

  def __init__(self, url="https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/1") -> None:
    super().__init__()
    print("initialize graph.")


    # Graph set up.
    self.g = tf.Graph()
    with self.g.as_default():
      self.text_input = tf.placeholder(dtype=tf.string, shape=[None])
      self.embed = hub.Module(url)
      self.embedded_text = self.embed(self.text_input)
      self.init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
    self.g.finalize()

    self.session = tf.Session(graph=self.g)
    self.session.run(self.init_op)


  def get_embedding(self, texts):
    # Reduce logging output.
    # tf.logging.set_verbosity(tf.logging.ERROR)

    with tf.Session(graph=self.g) as session:
      session.run(self.init_op)
      texts_embeddings = session.run(self.embedded_text, feed_dict={self.text_input: texts})
      # for i, message_embedding in enumerate(np.array(texts_embeddings).tolist()):
      #   print("Message: {}".format(texts[i]))
      #   print("Embedding size: {}".format(len(message_embedding)))
      #   message_embedding_snippet = ", ".join(
      #     (str(x) for x in message_embedding[:3]))
      #   print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

      return texts_embeddings

  def get_embedding_wo_session(self, texts):
    texts_embeddings = self.session.run(self.embedded_text, feed_dict={self.text_input: texts})

    return texts_embeddings

  def close(self):
    print('TFHubContext closed')


class UniversalSentenceEncodeMultilingual:
  default_context = TFHubContext(url="https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/1")