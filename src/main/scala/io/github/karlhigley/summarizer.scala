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

case class Sentence(text: String, features: Vector)

object Summarizer extends Logging {
  type Vertex = Tuple2[Long, Sentence]

  val hashingTF = new HashingTF()

  def dot(v1: Vector, v2: Vector) : Double = {
    BDV(v1.toArray).dot(BDV(v2.toArray))
  }

  def tokenize(text: String) : Seq[String] = {
    val tokenizer = SimpleEnglishTokenizer()
    tokenizer(text.trim.toLowerCase).toSeq    
  }

  def segment(text: String) : Seq[String] = {
    JavaSentenceSegmenter(text).toSeq
  }

  def featurizeSentences(rawSentences: RDD[String], stopwords: Set[String]) : RDD[Sentence] = {
    val sentencesWithTFs = rawSentences.map(s => {
      val tokens = tokenize(s).filter(!stopwords.contains(_))
      (s, hashingTF.transform(tokens))
    })
    
    val idf = new IDF().fit(sentencesWithTFs.map(_._2))    
    val normalizer = new Normalizer()    

    sentencesWithTFs.map( swtf => {      
      val featureVector = normalizer.transform(idf.transform(swtf._2))
      new Sentence(swtf._1, featureVector)
    })
  }

  def buildEdges(vertices: RDD[Vertex]) : RDD[Edge[Double]] = {
    vertices
      .cartesian(vertices)
      .filter(pair => pair._1._2.text != pair._2._2.text)
      .distinct()
      .map( pair => Edge(pair._1._1, pair._2._1, dot(pair._1._2.features, pair._2._2.features)) )
  }

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Summarizer")
    val sc   = new SparkContext(conf)

    val stopwords = Source.fromFile("stopwords.txt").getLines.toSet
    sc.broadcast(stopwords)

    val rawSentences = sc.textFile("test.txt").flatMap(segment)
    
    val vertices = featurizeSentences(rawSentences, stopwords).zipWithIndex().map(v => (v._2, v._1))
    val edges    = buildEdges(vertices)

    val sentenceGraph: Graph[Sentence, Double] = Graph(vertices, edges)

    val ranks = sentenceGraph
                  .subgraph(epred = (edge) => edge.attr >= 0.1)
                  .pageRank(0.0001)
                  .vertices

    val sentencesByRank = vertices
                            .join(ranks)
                            .map { case (id, (sentence, rank)) => (rank, sentence.text) }
                            .sortByKey(false)

    sentencesByRank.saveAsTextFile("ranked-sentences")

    sc.stop()
  }
}
