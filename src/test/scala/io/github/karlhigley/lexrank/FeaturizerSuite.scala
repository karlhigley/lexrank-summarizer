package io.github.karlhigley.lexrank

import org.scalatest.FunSuite

class FeaturizerSuite extends FunSuite with TestSparkContext {
  val allSentences = List(
    SentenceTokens(1L, "doc1", List("one", "two", "three")),
    SentenceTokens(2L, "doc1", List("three", "four", "five")),
    SentenceTokens(3L, "doc1", List("five", "six", "one", "three")),
    SentenceTokens(4L, "doc1", List("alpha", "beta", "gamma")),
    SentenceTokens(5L, "doc1", List())
  )

  val featurizer = new Featurizer

  test("feature vectors include only non-zero entries") {
    val sentences = sc.parallelize(allSentences)
    val featurized = featurizer(sentences).collect()
    featurized.foreach { f =>
        assert(!f.features.values.contains(0.0))
    }
  }

  test("single occurrence tokens are ignored") {
    val sentences = sc.parallelize(allSentences)
    val featurized = featurizer(sentences).collect()
    assert(featurized(0).features.indices.size === 2)
  }

  test("empty/zero vectors are omitted") {
    val sentences = sc.parallelize(allSentences)
    val featurized = featurizer(sentences).collect()
    assert(featurized.length === 3)
  }

  test("stopwords are removed") {
    val stopwordsFeaturizer = new Featurizer(numStopwords = 1)
    
    val sentences = sc.parallelize(allSentences)
    val featurized = stopwordsFeaturizer(sentences).collect()

    val stopwordIndex = stopwordsFeaturizer.indexOf("three")
    featurized.foreach { f =>
        assert(!f.features.indices.contains(stopwordIndex))
    }
  }
}