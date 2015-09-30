package io.github.karlhigley.lexrank

import scala.math.max

import org.scalatest.FunSuite

class DocumentSegmenterSuite extends FunSuite with TestSparkContext {
  val doc1 = """
    Apache Spark is an open-source cluster computing framework originally developed in the AMPLab at UC Berkeley.
    In contrast to Hadoop's two-stage disk-based MapReduce paradigm, Spark's in-memory primitives provide performance up to 100 times faster for certain applications.
    By allowing user programs to load data into a cluster's memory and query it repeatedly, Spark is well suited to machine learning algorithms.
  """

  val doc2 = """
    Spark requires a cluster manager and a distributed storage system.
    For cluster management, Spark supports standalone (native Spark cluster), Hadoop YARN, or Apache Mesos.
    For distributed storage, Spark can interface with a wide variety, including Hadoop Distributed File System (HDFS), Cassandra, OpenStack Swift, and Amazon S3.
    Spark also supports a pseudo-distributed local mode, usually used only for development or testing purposes, where distributed storage is not required and the local file system can be used instead; in this scenario, Spark is running on a single machine with one executor per CPU core.
  """

  val doc3 = """
    Spark had over 465 contributors in 2014, making it the most active project in the Apache Software Foundation and among Big Data open source projects.
  """

  val segmenter = new DocumentSegmenter

  val localDocs = List(doc1, doc2, doc3).zipWithIndex.map({ case (text, id) => Document(id.toString, text) })

  test("sentences are segmented reasonably") {    
    val documents = sc.parallelize(localDocs)
    val (sentences, tokenized) = segmenter(documents)
    assert(sentences.count() === 8)
  }

  test("tokens are alphabetic and lowercase") {
    val documents = sc.parallelize(localDocs)
    val (sentences, tokenized) = segmenter(documents)
    val tokens = tokenized.flatMap(_.tokens).collect()
    tokens.foreach { t =>
      assert("^[a-z]*$".r.findFirstIn(t).isEmpty === false)
    }
  }

  test("short tokens are removed") {
    val documents = sc.parallelize(localDocs)
    val (sentences, tokenized) = segmenter(documents)
    val tokens = tokenized.flatMap(_.tokens).collect()
    List("is", "an", "the", "at").foreach { s =>
      assert(tokens.find(_ == s).isEmpty === true)
    }
  }

}