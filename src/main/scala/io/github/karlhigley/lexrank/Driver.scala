package io.github.karlhigley.lexrank

import scala.io.Source

import org.apache.spark.{SparkContext, SparkConf, Logging}
import org.apache.spark.rdd.RDD

import org.apache.spark.graphx._

object Driver extends Logging {
  val serializerClasses: Array[Class[_]] = Array(
    classOf[Document], classOf[Sentence],
    classOf[SentenceTokens], classOf[SentenceFeatures],
    classOf[Featurizer], classOf[SignRandomProjectionLSH],
    classOf[LexRank]
  )

  private def selectExcerpts(sentences: RDD[Sentence], scores: VertexRDD[Double], length: Int) = {
    scores
      .join(sentences.map(s => (s.id, s)))
      .map { case (_, (score, sentence)) => (sentence.docId, (score, sentence.id, sentence.text)) }
      .groupByKey()
      .flatMap { case (docId, sentences) => sentences.toSeq.sortWith(_._1 > _._1).take(length).map(e => (docId, e._3)) }
  }

  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("Summarizer")
    sparkConf.registerKryoClasses(serializerClasses)

    val sc        = new SparkContext(sparkConf)
    val config    = new Configuration(args)

    val stopwords = Source.fromFile(config.stopwordsPath).getLines.toSet
    sc.broadcast(stopwords)
    
    val documents = sc.textFile(config.inputPath, minPartitions = config.partitions).flatMap( 
      _.split('\t').toList match {
        case List(docId, text @ _*) => Some((docId, text.mkString(" ")))
        case _                 => None
      }
    ).map(Document.tupled)

    val segmenter = new DocumentSegmenter(stopwords)
    val (sentences, tokenized) = segmenter(documents)

    val featurizer = new Featurizer
    val features = featurizer(tokenized)

    val comparer = new SimilarityComparison(config.threshold, config.buckets)
    val comparisons = comparer(features)

    val model    = LexRank.build(features, comparisons)
    val ranks    = model.score(config.cutoff, config.convergence)
    val excerpts = selectExcerpts(sentences, ranks, config.length)

    excerpts
      .map(_.productIterator.toList.mkString("\t"))
      .saveAsTextFile(config.outputPath)

    sc.stop()
  }
}