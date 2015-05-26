package io.github.karlhigley.lexrank

import org.scalatest.FunSuite

class FeaturizerSuite extends FunSuite with TestSparkContext {
  val sentence1 = SentenceTokens(1L, "doc1", List("one", "two", "three"))
  val sentence2 = SentenceTokens(2L, "doc1", List("three", "four", "five"))
  val sentence3 = SentenceTokens(3L, "doc1", List("five", "six", "one"))

  val featurizer = new Featurizer

  test("single occurrence tokens are ignored") {
    val sentences = sc.parallelize(List(sentence1, sentence2, sentence3))
    val featurized = featurizer(sentences).collect()
    assert(featurized(0).features.indices.size === 2)
  }

  test("zero vectors are omitted") {
    val sentences = sc.parallelize(List(sentence1))
    val featurized = featurizer(sentences).collect()
    assert(featurized.length === 0)
  }

  test("empty vectors are omitted") {
    val sentence = SentenceTokens(4L, "doc1", List())
    val sentences = sc.parallelize(List(sentence))
    val featurized = featurizer(sentences).collect()
    assert(featurized.length === 0)
  }
}