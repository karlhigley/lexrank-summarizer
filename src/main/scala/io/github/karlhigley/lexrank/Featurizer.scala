package io.github.karlhigley.lexrank

import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.feature.{HashingTF, IDF}
import org.apache.spark.mllib.linalg.SparseVector

import chalk.text.analyze.PorterStemmer
import chalk.text.segment.JavaSentenceSegmenter
import chalk.text.tokenize.SimpleEnglishTokenizer

case class Document(id: String, text: String)
case class Sentence(id: Long, docId: String, text: String)
case class SentenceTokens(id: Long, docId: String, tokens: Seq[String])
case class SentenceFeatures(id: Long, docId: String, features: SparseVector)

class Featurizer(stopwords: Set[String]) extends Serializable {
  def featurize(documents: RDD[Document]) = {  
    val sentences = extractSentences(documents)
    val tokenized = tokenize(sentences, stopwords)
    val features  = vectorize(tokenized).filter(f => f.features.indices.size > 0)
    (sentences, features)
  }

  private def extractSentences(documents: RDD[Document]) : RDD[Sentence] = {
    documents
      .flatMap(d => segment(d.text).map(t => (d.id, t)) )
      .zipWithIndex()
      .map({
        case ((docId, sentenceText), sentenceId) => Sentence(sentenceId, docId, sentenceText)
      })
  }

  private def tokenize(sentences: RDD[Sentence], stopwords: Set[String]) : RDD[SentenceTokens] = {
    val tokenizer = SimpleEnglishTokenizer()
    sentences.map(s => {
      val tokens = tokenizer(s.text.toLowerCase).toSeq.filter(!stopwords.contains(_)).map(stem)
      SentenceTokens(s.id, s.docId, tokens)
    })
  }

  private def vectorize(tokens: RDD[SentenceTokens]) : RDD[SentenceFeatures] = {
    val hashingTF  = new HashingTF()
    val idfModel   = new IDF()

    val termFrequencies = tokens.map(t => {
        (t.id, t.docId, hashingTF.transform(t.tokens))
    })
    
    val idf = idfModel.fit(termFrequencies.map({ case (_, _, tf) => tf }))

    termFrequencies.map({
      case (id, docId, tf) =>
        val featureVector = idf.transform(tf).asInstanceOf[SparseVector]
        SentenceFeatures(id, docId, featureVector)
    })
  }

  private def segment(text: String) : Seq[String] = {
    JavaSentenceSegmenter(text).toSeq
  }

  private def stem(token: String) : String = {
    PorterStemmer(token)
  }
}

  