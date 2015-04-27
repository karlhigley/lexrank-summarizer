package io.github.karlhigley.lexrank

import scala.io.Source

import org.apache.spark.{SparkContext, SparkConf, Logging}
import org.apache.spark.rdd.RDD

import org.apache.spark.graphx._

object Driver extends Logging {
  private def selectExcerpts(sentences: RDD[Sentence], scores: VertexRDD[Double], length: Int) = {
    scores
      .join(sentences.map(s => (s.id, s)))
      .map { case (_, (score, sentence)) => (sentence.docId, (score, sentence.id, sentence.text)) }
      .groupByKey()
      .flatMap { case (docId, sentences) => sentences.toSeq.sortWith(_._1 > _._1).take(length).map(e => (docId, e._3)) }
  }

  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("Summarizer")
    val sc        = new SparkContext(sparkConf)

    val config    = new Configuration(args)

    val stopwords = Source.fromFile(config.stopwordsPath).getLines.toSet
    sc.broadcast(stopwords)
    val featurizer = new Featurizer(stopwords)

    val documents = sc.textFile(config.inputPath).flatMap( 
      _.split('\t').toList match {
        case List(docId, text) => Some(Document(docId, text))
        case _                 => None
      }
    )

    val (sentences, features) = featurizer.featurize(documents)

    val model    = new LexRank(features)
    val ranks    = model.score(config.threshold, config.cutoff, config.convergence)
    val excerpts = selectExcerpts(sentences, ranks, config.length)

    excerpts
      .map(_.productIterator.toList.mkString("\t"))
      .saveAsTextFile(config.outputPath)

    sc.stop()
  }
}