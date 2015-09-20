package io.github.karlhigley.lexrank

import scala.collection.immutable.BitSet
import scala.math

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

import org.apache.spark.graphx._


class LexRank(graph: Graph[String, Float]) extends Serializable {
  def score(cutoff: Float = 0.8f, convergence: Float = 0.001f) : VertexRDD[Double] = {
    filterBoilerplate(graph, cutoff).pageRank(convergence).vertices
  }

  private def filterBoilerplate(graph: Graph[String, Float], cutoff: Float): Graph[String, Float] = {
    val maxSimilarityVertices = Graph.fromEdges(graph.edges, 0).aggregateMessages[Float](
      sendMsg       = triplet => triplet.sendToDst(triplet.attr),
      mergeMsg      = (a, b) => math.max(a,b),
      tripletFields = TripletFields.EdgeOnly
    )

    val maxSimilarityGraph: Graph[Float, Float] = Graph(maxSimilarityVertices, graph.edges, 1.0f)
    val filteredGraph = maxSimilarityGraph.subgraph(vpred = (id, attr) => attr < cutoff)

    graph.mask(filteredGraph).subgraph(
      epred = (triplet)   => triplet.srcAttr == triplet.dstAttr,
      vpred = (id, attr)  => attr != ""
    )
  }
}

object LexRank {
  val SIGNATURE_BITS = 256
  val FREQUENCY = 1.0e-5

  def build(featurized: RDD[SentenceFeatures], storageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK_SER): LexRank = {
    featurized.persist(storageLevel)
    val numSentences = featurized.count()

    val signatures = generateSignatures(featurized)
    val remainingSentences = filterOutCommonSignatures(signatures, numSentences)
    remainingSentences.persist(storageLevel)
    remainingSentences.count()

    featurized.unpersist()

    val vertices = generateVertices(remainingSentences)
    vertices.persist(storageLevel)
    vertices.count()

    val edges = generateEdges(remainingSentences)
    edges.persist(storageLevel)
    edges.count()

    remainingSentences.unpersist()

    val graph = Graph(vertices, edges)
    new LexRank(graph)
  }

  private def generateSignatures(featurized: RDD[SentenceFeatures]) = {
    val signatureGen = new SignRandomProjectionLSH  

    featurized.map(f => {
      (signatureGen.computeSignature(f.features, SIGNATURE_BITS), f)
    })
  }

  private def filterOutCommonSignatures(sentencesBySignature: RDD[Tuple2[BitSet, SentenceFeatures]], numSentences: Long): RDD[Tuple2[BitSet, SentenceFeatures]] = {
    val countsBySignature = sentencesBySignature.map(s => (s._1, 1)).reduceByKey(_ + _)

    val boilerplateSignatures = countsBySignature.flatMap {
        case (sig, count) if count > math.max(1, numSentences * FREQUENCY) => Some(sig)
        case _ => None
    }.collect()

    val brBoilerplateSignatures = sentencesBySignature.context.broadcast(boilerplateSignatures)
    
    sentencesBySignature.filter(pair => !brBoilerplateSignatures.value.contains(pair._1))
  }

  private def generateVertices(sentencesBySignature: RDD[Tuple2[BitSet, SentenceFeatures]]) = {
    sentencesBySignature.map {
      case (signature, features) => (features.id, features.docId)
    }
  }

  private def generateEdges(sentencesBySignature: RDD[Tuple2[BitSet, SentenceFeatures]]): RDD[Edge[Float]] = {
    val byDocId = sentencesBySignature.map {
      case (signature, features) => (features.docId, (features.id, signature))
    }.persist()

    val unidirectionalEdges = byDocId.join(byDocId).map(pair => pair._2).flatMap {
      case ((idA, sigA), (idB, sigB)) =>
        val estimate = SignRandomProjectionLSH.estimateCosine(sigA, sigB, SIGNATURE_BITS).toFloat
        if (estimate > 0.1 && idA != idB) {
          Some(Edge[Float](idA, idB, estimate))  
        } else {
          None
        }
      case _ => None
    }.repartition(sentencesBySignature.partitions.length).persist()

    unidirectionalEdges.count()
    byDocId.unpersist()

    val bidirectionalEdges = unidirectionalEdges.flatMap( edge => List(edge, Edge[Float](edge.dstId, edge.srcId, edge.attr)))

    bidirectionalEdges
  }
}
