package io.github.karlhigley

import scala.io.Source

import org.apache.spark.{SparkContext, SparkConf, Logging}
import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.feature.{HashingTF, IDF, Normalizer}
import org.apache.spark.mllib.linalg.{Vector}

import org.apache.spark.graphx._

import breeze.linalg.{DenseVector => BDV}

import chalk.text.segment.JavaSentenceSegmenter
import chalk.text.tokenize.SimpleEnglishTokenizer

case class Sentence(id: Long, text: String)
case class TokenizedSentence(id: Long, tokens: Seq[String])
case class FeaturizedSentence(id: Long, features: Vector)

object Summarizer extends Logging {
  def dot(v1: Vector, v2: Vector) : Double = {
    BDV(v1.toArray).dot(BDV(v2.toArray))
  }

  def segment(text: String) : Seq[String] = {
    JavaSentenceSegmenter(text).toSeq
  }

  def extractSentences(documents: RDD[String]) : RDD[Sentence] = {
    // For now, only expect one document so flatMap instead of map
    documents.flatMap(segment(_))
             .zipWithIndex()
             .map({ case (text, id) => Sentence(id, text) })
  }

  def tokenize(sentences: RDD[Sentence], stopwords: Set[String]) : RDD[TokenizedSentence] = {
    val tokenizer = SimpleEnglishTokenizer()
    sentences.map(s => {
      val tokens = tokenizer(s.text.toLowerCase).toSeq.filter(!stopwords.contains(_))
      TokenizedSentence(s.id, tokens)
    })
  }

  def featurize(tokenizedSentences: RDD[TokenizedSentence]) : RDD[FeaturizedSentence] = {
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
        FeaturizedSentence(id, featureVector)
    })
  }

  def buildEdges(vertices: RDD[(Long, Vector)]) : RDD[Edge[Double]] = {
    vertices
      .cartesian(vertices)
      .filter({ case (v1, v2) => v1 != v2 })
      .distinct()
      .map({
        case ((id1, features1), (id2, features2)) =>
          Edge(id1, id2, dot(features1, features2)) 
      })
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Summarizer")
    val sc   = new SparkContext(conf)

    val stopwords = Source.fromFile("stopwords.txt").getLines.toSet
    sc.broadcast(stopwords)

    val sentences          = extractSentences(sc.textFile("test.txt"))
    val tokenizedSentences = tokenize(sentences, stopwords)
    val vertices           = featurize(tokenizedSentences).map(fs => (fs.id, fs.features))
    val edges              = buildEdges(vertices)

    val sentenceGraph: Graph[Vector, Double] = Graph(vertices, edges)

    val ranks = sentenceGraph
                  .subgraph(epred = (edge) => edge.attr >= 0.1)
                  .pageRank(0.0001)
                  .vertices

    val sentencesByRank = vertices
                            .join(ranks)
                            .map { case (id, (features, rank)) => (id, rank) }
                            .join(sentences.map(s => (s.id, s.text)))
                            .map { case (id, (rank, text)) => (rank, text) }
                            .sortByKey(false)

    sentencesByRank.saveAsTextFile("ranked-sentences")

    sc.stop()
  }
}
