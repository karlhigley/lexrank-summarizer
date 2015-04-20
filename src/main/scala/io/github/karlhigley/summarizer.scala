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
case class SentenceTokens(id: Long, docId: String, tokens: Seq[String])
case class SentenceFeatures(id: Long, docId: String, features: Vector)

class LexRank(sentences: RDD[Sentence], features: RDD[SentenceFeatures]) extends Serializable {
  def summarize() = {
    val graph  = buildGraph(features, 0.1)
    val scores = graph.pageRank(0.0001).vertices   
    selectExcerpts(sentences, scores)   
  }

  private def dot(v1: Vector, v2: Vector) : Double = {
    BDV(v1.toArray).dot(BDV(v2.toArray))
  }

  private def buildGraph(features: RDD[SentenceFeatures], threshold: Double): Graph[Vector, Double] = {
    val edges =
      features
        .cartesian(features)
        .filter({ case (v1, v2) => v1 != v2 })
        .distinct()
        .flatMap({
          case (a: SentenceFeatures, b: SentenceFeatures) =>
            dot(a.features, b.features) match {
              case similarity if similarity > threshold => Some(Edge(a.id, b.id, similarity))
              case _ => None
            }
        })
    val vertices = features.map(f => (f.id, f.features))
    Graph(vertices, edges)
  }

  private def selectExcerpts(sentences: RDD[Sentence], ranks: VertexRDD[Double]) = {
    ranks
      .join(sentences.map(s => (s.id, s)))
      .map { case (_, (rank, sentence)) => (sentence.docId, (rank, sentence.id, sentence.text)) }
      .groupByKey()
      .flatMap { case (docId, sentences) => sentences.toSeq.sortWith(_._1 > _._1).take(5).map(e => (docId, e._3)) }
  }
}

object LexRank {
  def featurize(documents: RDD[Document], stopwords: Set[String]) = {  
    val sentences = extractSentences(documents)
    val tokenized = tokenize(sentences, stopwords)
    val features  = vectorize(tokenized)
    new LexRank(sentences, features)
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
    val normalizer = new Normalizer()
    val idfModel   = new IDF()

    val termFrequencies = tokens.map(t => {
        SentenceFeatures(t.id, t.docId, hashingTF.transform(t.tokens))
    })
    
    val idf = idfModel.fit(termFrequencies.map({ case SentenceFeatures(_, _, tf) => tf }))

    termFrequencies.map({
      case SentenceFeatures(id, docId, tf) =>
        val featureVector = normalizer.transform(idf.transform(tf))
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

class Configuration(args: Array[String]) {
  var inputPath     = "input"
  var outputPath    = "output"
  var stopwordsPath = "stopwords"

  parse(args.toList)

  private def parse(args: List[String]): Unit = args match {
    case ("--input" | "-i") :: path :: tail =>
      inputPath = path
      parse(tail)

    case ("--output" | "-o") :: path :: tail =>
      outputPath = path
      parse(tail)

    case ("--stopwords" | "-s") :: path :: tail =>
      stopwordsPath = path
      parse(tail)

    case ("--help" | "-h") :: tail =>
      printUsageAndExit(0)

    case _ =>
  }

  /**
   * Print usage and exit JVM with the given exit code.
   */
  private def printUsageAndExit(exitCode: Int) {
    val usage =
     s"""
      |Usage: spark-submit --class io.github.karlhigley.Summarizer <jar-path> [summarizer options]
      |
      |Options:
      |   -i PATH, --input PATH          Relative path of input files (default: "./input")
      |   -o PATH, --output PATH         Relative path of output files (default: "./output")
      |   -s PATH, --stopwords PATH      Relative path of stopwords file (default: "./stopwords")
     """.stripMargin
    System.err.println(usage)
    System.exit(exitCode)
  }

}

object Summarizer extends Logging {
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
    val excerpts = model.summarize()

	excerpts
      .map(_.productIterator.toList.mkString("\t"))
      .saveAsTextFile(config.outputPath)

    sc.stop()
  }
}
