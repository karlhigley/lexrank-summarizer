package io.github.karlhigley.lexrank

import scala.math.max

import org.apache.spark.rdd.RDD

import org.apache.spark.mllib.feature.{HashingTF, IDF, Normalizer}
import org.apache.spark.mllib.linalg.{Vector, SparseVector}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, RowMatrix}

import org.apache.spark.graphx._

import chalk.text.analyze.PorterStemmer
import chalk.text.segment.JavaSentenceSegmenter
import chalk.text.tokenize.SimpleEnglishTokenizer

case class Document(id: String, text: String)
case class Sentence(id: Long, docId: String, text: String)
case class SentenceTokens(id: Long, docId: String, tokens: Seq[String])
case class SentenceFeatures(id: Long, docId: String, features: SparseVector)
case class SentenceComparison(id1: Long, id2: Long, similarity: Double)

class LexRank(sentences: RDD[Sentence], features: RDD[SentenceFeatures]) extends Serializable {
  def summarize(length: Int = 5, threshold: Double = 0.1, cutoff: Double = 0.8, convergence: Double = 0.001) = {    
    val graph        = buildGraph(features, threshold, cutoff)
    val scores       = graph.pageRank(convergence).vertices   
    selectExcerpts(sentences, scores, length)   
  }

  private def buildGraph(features: RDD[SentenceFeatures], threshold: Double, cutoff: Double): Graph[String, Double] = {
    val comparisons         = compareSentences(features, threshold)
    val filteredGraph       = removeBoilerplate(comparisons, cutoff)

    val docIds              = features.map(f => (f.id, f.docId))
    val documentComponents  = removeCrossDocEdges(filteredGraph, docIds)

    documentComponents
  }

  private def removeBoilerplate(comparisons: RDD[SentenceComparison], boilerplateCutoff: Double): Graph[Double, Double] = {
    val edges = comparisons
                  .flatMap(c => SentenceComparison.unapply(c))
                  .map(e => Edge(e._1, e._2, e._3))

    val maxSimilarityVertices = Graph.fromEdges(edges, 0).aggregateMessages[Double](
      sendMsg       = triplet => triplet.sendToDst(triplet.attr),
      mergeMsg      = (a, b) => max(a,b),
      tripletFields = TripletFields.EdgeOnly
    )

    val maxSimilarityGraph: Graph[Double, Double] = Graph(maxSimilarityVertices, edges, 1.0)
    maxSimilarityGraph.subgraph(vpred = (id, attr) => attr < boilerplateCutoff)  
  }

  private def removeCrossDocEdges(graph: Graph[Double, Double], docIds: RDD[(Long, String)]): Graph[String, Double] = {
    val docGraph: Graph[String, Double] = graph.outerJoinVertices(docIds)(
      (id, similarity, docIdOpt) => docIdOpt match {
        case Some(docId)  => docId
        case None         => ""
      }
    )
    docGraph.subgraph(
      epred = (triplet)   => triplet.srcAttr == triplet.dstAttr,
      vpred = (id, attr)  => attr != ""
    )
  }

  private def selectExcerpts(sentences: RDD[Sentence], ranks: VertexRDD[Double], length: Int) = {
    ranks
      .join(sentences.map(s => (s.id, s)))
      .map { case (_, (rank, sentence)) => (sentence.docId, (rank, sentence.id, sentence.text)) }
      .groupByKey()
      .flatMap { case (docId, sentences) => sentences.toSeq.sortWith(_._1 > _._1).take(length).map(e => (docId, e._3)) }
  }

  private def compareSentences(columns: RDD[SentenceFeatures], threshold: Double): RDD[SentenceComparison] = {
    buildRowMatrix(columns)
      .columnSimilarities(threshold)
      .entries
      .flatMap(MatrixEntry.unapply(_))
      .map(SentenceComparison.tupled)
      .filter(_.similarity > threshold)
  }

  private def buildRowMatrix(columns: RDD[SentenceFeatures]) : RowMatrix = {   
    val matrixEntries = columns.flatMap {
      case SentenceFeatures(colNum, _, vector) =>
        sparseElements(vector).map {
          case (rowNum, value) => MatrixEntry(rowNum, colNum, value)
        }
    }

    new CoordinateMatrix(matrixEntries).toRowMatrix()
  }

  private def sparseElements(vector: SparseVector): Seq[(Int, Double)] = {
    vector.indices.zip(vector.values)
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