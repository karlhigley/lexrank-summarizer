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

class DocumentSegmenter(stopwords: Set[String]) extends Serializable {
  def apply(documents: RDD[Document]) = {  
    val sentences = extractSentences(documents)
    val tokenized = tokenize(sentences, stopwords)
    (sentences, tokenized)
  }

  private def extractSentences(documents: RDD[Document]) : RDD[Sentence] = {
    documents
      .flatMap(d => segment(d.text).map(t => (d.id, t)) )
      .distinct()
      .zipWithIndex()
      .map({
        case ((docId, sentenceText), sentenceId) => Sentence(sentenceId, docId, sentenceText)
      })
  }

  private def tokenize(sentences: RDD[Sentence], stopwords: Set[String]) : RDD[SentenceTokens] = {
    val tokenizer = SimpleEnglishTokenizer()
    val nonWord   = "[^a-z]*".r

    sentences.map(s => {
      val tokens = tokenizer(s.text.toLowerCase).toSeq
                                          .map(nonWord.replaceAllIn(_, ""))
                                          .filter(_.length > 3)
                                          .filter(!stopwords.contains(_))
                                          .map(stem)

      SentenceTokens(s.id, s.docId, tokens)
    })
  }

  private def segment(text: String) : Seq[String] = {
    JavaSentenceSegmenter(text).toSeq
  }

  private def stem(token: String) : String = {
    PorterStemmer(token)
  }
}

  