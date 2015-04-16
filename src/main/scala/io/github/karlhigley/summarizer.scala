package io.github.karlhigley

import scala.io.Source

import org.apache.spark.{SparkContext, SparkConf, Logging}
import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.feature.{HashingTF, IDF, Normalizer}
import org.apache.spark.mllib.linalg.{Vector}

import org.apache.spark.graphx._

import breeze.linalg.{DenseVector => BDV}

import chalk.text.analyze.PorterStemmer
import chalk.text.segment.JavaSentenceSegmenter
import chalk.text.tokenize.SimpleEnglishTokenizer

case class Document(id: String, text: String)
case class Sentence(id: Long, docId: String, text: String)
case class TokenizedSentence(id: Long, tokens: Seq[String])

class LexRank(stopwords: Set[String]) extends Serializable {
  def summarize(documents: RDD[Document]) = {
    val sentences           = extractSentences(documents)
    val tokenizedSentences  = tokenize(sentences, stopwords)
    val featurizedSentences = featurize(tokenizedSentences)
    val ranks               = rankSentences(featurizedSentences)
    selectExcerpts(sentences, ranks)   
  }

  private def dot(v1: Vector, v2: Vector) : Double = {
    BDV(v1.toArray).dot(BDV(v2.toArray))
  }

  private def segment(text: String) : Seq[String] = {
    JavaSentenceSegmenter(text).toSeq
  }

  private def stem(token: String) : String = {
    PorterStemmer(token)
  }

  private def extractSentences(documents: RDD[Document]) : RDD[Sentence] = {
    documents
      .flatMap(d => segment(d.text).map(t => (d.id, t)) )
      .zipWithIndex()
      .map({
        case ((docId, sentenceText), sentenceId) => Sentence(sentenceId, docId, sentenceText)
      })
  }

  private def tokenize(sentences: RDD[Sentence], stopwords: Set[String]) : RDD[TokenizedSentence] = {
    val tokenizer = SimpleEnglishTokenizer()
    sentences.map(s => {
      val tokens = tokenizer(s.text.toLowerCase).toSeq.filter(!stopwords.contains(_)).map(stem)
      TokenizedSentence(s.id, tokens)
    })
  }

  private def featurize(tokenizedSentences: RDD[TokenizedSentence]) : RDD[Tuple2[Long, Vector]] = {
    val hashingTF  = new HashingTF()
    val normalizer = new Normalizer()
    val idfModel   = new IDF()

    val termFrequencies = tokenizedSentences.map(s => {
        (s.id, hashingTF.transform(s.tokens))
    })
    
    val idf = idfModel.fit(termFrequencies.map({ case (id, tf) => tf }))

    termFrequencies.map({
      case (id, tf) =>
        val featureVector = normalizer.transform(idf.transform(tf))
        (id, featureVector)
    })
  }

  private def buildEdges(vertices: RDD[(Long, Vector)]) : RDD[Edge[Double]] = {
    vertices
      .cartesian(vertices)
      .filter({ case (v1, v2) => v1 != v2 })
      .distinct()
      .map({
        case ((id1, features1), (id2, features2)) =>
          Edge(id1, id2, dot(features1, features2)) 
      })
  }

  private def rankSentences(featurizedSentences: RDD[Tuple2[Long, Vector]]) : VertexRDD[Double] = {
    val edges         = buildEdges(featurizedSentences)
    val sentenceGraph = Graph(featurizedSentences, edges)
    sentenceGraph
      .subgraph(epred = (edge) => edge.attr >= 0.1)
      .pageRank(0.0001)
      .vertices    
  }

  private def selectExcerpts(sentences: RDD[Sentence], ranks: VertexRDD[Double]) = {
    ranks
      .join(sentences.map(s => (s.id, s)))
      .map { case (sentenceId, (rank, sentence)) => (sentence.docId, (rank, sentence.id, sentence.text)) }
      .groupByKey()
      .flatMap { case (docId, sentences) => sentences.toSeq.sortWith(_._1 > _._1).take(5).map(e => (docId, e._3)) }
  }

}

object Summarizer extends Logging {
  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Summarizer")
    val sc   = new SparkContext(conf)

    val stopwords = Source.fromFile("stopwords").getLines.toSet
    sc.broadcast(stopwords)

    val documents = sc.textFile("input").flatMap( 
      _.split('\t').toList match {
        case List(docId, text) => Some(Document(docId, text))
        case _                 => None
      }
    )

    val model    = new LexRank(stopwords)
    val excerpts = model.summarize(documents)

	excerpts
      .map(_.productIterator.toList.mkString("\t"))
      .saveAsTextFile("output")

    sc.stop()
  }
}
