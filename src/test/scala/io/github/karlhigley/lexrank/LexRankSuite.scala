package io.github.karlhigley.lexrank

import org.scalatest.FunSuite

import org.apache.spark.mllib.linalg.SparseVector

case class SentenceFeatures(id: Long, docId: String, features: SparseVector)

class LexRankSuite extends FunSuite with TestSparkContext {
  val indices1 = (0 until 10).toArray
  val indices2 = indices1.map(_ + 1)
  val indices3 = indices1.map(_ + 5)
  
  val features = Array.fill(indices1.length)(0.5)

  val feature1 = SentenceFeatures(1L, "doc1", new SparseVector(100, indices1, features))
  val feature2 = SentenceFeatures(2L, "doc2", new SparseVector(100, indices2, features))
  val feature3 = SentenceFeatures(3L, "doc2", new SparseVector(100, indices3, features))

  test("identical vertices are removed") {
    val features = sc.parallelize(List(feature1, feature1))
    val lexrank = LexRank.build(features)

    val scores = lexrank.score(cutoff = 0.8f)

    assert(scores.count() === 0)
  }

  test("vertices above similarity cutoff are removed") {
    val features = sc.parallelize(List(feature1, feature2))
    val lexrank = LexRank.build(features)

    val scores = lexrank.score(cutoff = 0.8f)

    assert(scores.count() === 0)
  }

  test("vertices below similarity cutoff are retained") {
    val features = sc.parallelize(List(feature2, feature3))
    val lexrank = LexRank.build(features)

    val scores = lexrank.score(cutoff = 0.8f)

    assert(scores.count() === 2)
  }
}