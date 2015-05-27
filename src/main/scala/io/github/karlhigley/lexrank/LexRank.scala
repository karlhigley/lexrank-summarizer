package io.github.karlhigley.lexrank

import scala.math.max

import org.apache.spark.rdd.RDD
import org.apache.spark.graphx._

case class SentenceComparison(id1: Long, id2: Long, similarity: Double)

class LexRank(graph: Graph[String, Double]) extends Serializable {
  def score(cutoff: Double = 0.8, convergence: Double = 0.001) : VertexRDD[Double] = {
    filterBoilerplate(graph, cutoff).pageRank(convergence).vertices
  }

  private def filterBoilerplate(graph: Graph[String, Double], cutoff: Double): Graph[String, Double] = {
    val maxSimilarityVertices = Graph.fromEdges(graph.edges, 0).aggregateMessages[Double](
      sendMsg       = triplet => triplet.sendToDst(triplet.attr),
      mergeMsg      = (a, b) => max(a,b),
      tripletFields = TripletFields.EdgeOnly
    )

    val maxSimilarityGraph: Graph[Double, Double] = Graph(maxSimilarityVertices, graph.edges, 1.0)
    val filteredGraph = maxSimilarityGraph.subgraph(vpred = (id, attr) => attr < cutoff)

    graph.mask(filteredGraph).subgraph(
      epred = (triplet)   => triplet.srcAttr == triplet.dstAttr,
      vpred = (id, attr)  => attr != ""
    )
  }
}

object LexRank {
  def build(features: RDD[SentenceFeatures], comparisons: RDD[SentenceComparison]): LexRank = {
    val vertices = features.map(f => (f.id, f.docId))

    val edges = comparisons
                  .flatMap(c => SentenceComparison.unapply(c))
                  .flatMap(e => List(Edge(e._1, e._2, e._3), Edge(e._2, e._1, e._3)))    

    new LexRank(Graph(vertices, edges))
  }
}