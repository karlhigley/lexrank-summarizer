package io.github.karlhigley.lexrank

import org.scalatest.FunSuite

import org.apache.spark.mllib.linalg.SparseVector

case class SentenceFeatures(id: Long, docId: String, features: SparseVector)

class LexRankSuite extends FunSuite with TestSparkContext {
  val feature1 = SentenceFeatures(1L, "doc1", new SparseVector(100, Array(), Array()))
  val feature2 = SentenceFeatures(2L, "doc2", new SparseVector(100, Array(), Array()))
  val feature3 = SentenceFeatures(3L, "doc2", new SparseVector(100, Array(), Array()))

  test("vertices above similarity cutoff are removed") {
    val features = sc.parallelize(List(feature1, feature2))

    val comparison1 = SentenceComparison(1L, 2L, 0.9f)
    val comparisons = sc.parallelize(List(comparison1))

    val lexrank = LexRank.build(features, comparisons)

    val scores = lexrank.score(cutoff = 0.8f)

    assert(scores.count() === 0)
  }

  test("vertices below similarity cutoff are retained") {
    val features = sc.parallelize(List(feature2, feature3))

    val comparison1 = SentenceComparison(2L, 3L, 0.5f)
    val comparisons = sc.parallelize(List(comparison1))

    val lexrank = LexRank.build(features, comparisons)

    val scores = lexrank.score(cutoff = 0.8f)

    assert(scores.count() === 2)
  }
}