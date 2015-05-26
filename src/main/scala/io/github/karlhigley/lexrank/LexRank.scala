package io.github.karlhigley.lexrank

import scala.math.max

import org.apache.spark.rdd.RDD
import org.apache.spark.graphx._

case class SentenceComparison(id1: Long, id2: Long, similarity: Double)

class LexRank(features: RDD[SentenceFeatures], comparisons: RDD[SentenceComparison]) extends Serializable {
  def score(threshold: Double = 0.1, cutoff: Double = 0.8, convergence: Double = 0.001) = {    
    buildGraph(features, comparisons, threshold, cutoff).pageRank(convergence).vertices    
  }

  private def buildGraph(features: RDD[SentenceFeatures], comparisons: RDD[SentenceComparison], threshold: Double, cutoff: Double): Graph[String, Double] = {
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
}