package io.github.karlhigley.lexrank

import scala.io.Source

import org.apache.spark.{SparkContext, SparkConf, Logging}

object Driver extends Logging {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("Summarizer")
    val sc        = new SparkContext(sparkConf)

    val config    = new Configuration(args)

    val stopwords = Source.fromFile(config.stopwordsPath).getLines.toSet
    sc.broadcast(stopwords)

    val documents = sc.textFile(config.inputPath).flatMap( 
      _.split('\t').toList match {
        case List(docId, text) => Some(Document(docId, text))
        case _                 => None
      }
    )

    val model    = LexRank.featurize(documents, stopwords)
    val excerpts = model.summarize(config.length, config.threshold, config.cutoff, config.convergence)

  excerpts
      .map(_.productIterator.toList.mkString("\t"))
      .saveAsTextFile(config.outputPath)

    sc.stop()
  }
}