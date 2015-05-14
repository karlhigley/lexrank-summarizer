package io.github.karlhigley.lexrank

import scala.math.{log, max}

import org.apache.spark.rdd.RDD
import org.apache.spark.graphx._

import org.apache.spark.mllib.linalg.{Vector, SparseVector}
import org.apache.spark.mllib.linalg.distributed.{CoordinateMatrix, MatrixEntry, RowMatrix}

case class SentenceComparison(id1: Long, id2: Long, similarity: Double)

class LexRank(features: RDD[SentenceFeatures]) extends Serializable {
  def score(threshold: Double = 0.1, cutoff: Double = 0.8, convergence: Double = 0.001) = {    
    buildGraph(features, threshold, cutoff).pageRank(convergence).vertices    
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

  private def compareSentences(sentences: RDD[SentenceFeatures], threshold: Double): RDD[SentenceComparison] = {
    val matrices = bucketSentences(sentences).map(buildRowMatrix(_))
    matrices.foreach(_.rows.persist())

    val similarities = matrices
                          .map(computeSimilarities(_, threshold))
                          .reduce(_ union _)
                          .coalesce(sentences.partitions.size)

    similarities.persist()
    similarities.count()

    matrices.foreach(_.rows.unpersist())

    similarities.flatMap(MatrixEntry.unapply(_)).map(SentenceComparison.tupled)
  }

  private def bucketSentences(sentences: RDD[SentenceFeatures]) = {
    val signatureGen  = new CosineLSH
    val signatureBits = (log(sentences.partitions.size)/log(2)).toInt
    
    val signatures    = features.map(f => {
      (signatureGen.computeSignature(f.features, signatureBits), f)
    })
    
    CosineLSH
      .signatureSet(signatureBits)
      .map(k => {
        signatures.filter(s => s._1 == k).map(s => s._2)
      })
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

  private def computeSimilarities(matrix: RowMatrix, threshold: Double): RDD[MatrixEntry] = {
    matrix.columnSimilarities(threshold).entries.filter(_.value > threshold)
  }
}