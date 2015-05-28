package io.github.karlhigley.lexrank

import org.scalatest.FunSuite

import org.apache.spark.mllib.linalg.SparseVector

class SimilarityComparisonSuite extends FunSuite with TestSparkContext {
  val feature1 = SentenceFeatures(1L, "doc1", new SparseVector(100, Array(1,2,3), Array(1,1,1)))
  val feature2 = SentenceFeatures(2L, "doc1", new SparseVector(100, Array(4,5,6), Array(2,2,2)))
  val feature3 = SentenceFeatures(3L, "doc1", new SparseVector(100, Array(4,5,6), Array(2,2,2)))

  val comparer = new SimilarityComparison(threshold = 0.1, buckets = 2)

/*
  test("similarities below the threshold are omitted") {
    val features = sc.parallelize(List(feature1, feature2))
    val comparisons = comparer(features)

    assert(comparisons.count() === 0)
  }
*/

  test("similar pairs get high similarity score") {
    val features = sc.parallelize(List(feature2, feature3))
    val comparisons = comparer(features)

    assert(comparisons.first().similarity > 0.5)
  }
}