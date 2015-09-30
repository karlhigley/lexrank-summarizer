package io.github.karlhigley.lexrank

class Configuration(args: Array[String]) {
  var inputPath     = "input"
  var outputPath    = "output"

  var partitions    = 2
  var buckets       = 64
  var numStopwords  = 250
  var length        = 5
  var cutoff        = 0.8f
  var threshold     = 0.1f
  var convergence   = 0.001f

  parse(args.toList)

  private def parse(args: List[String]): Unit = args match {
    case ("--input" | "-i") :: path :: tail =>
      inputPath = path
      parse(tail)

    case ("--output" | "-o") :: path :: tail =>
      outputPath = path
      parse(tail)

    case ("--partitions" | "-p") :: value :: tail =>
      partitions = value.toInt
      parse(tail)

    case ("--buckets" | "-k") :: value :: tail =>
      buckets = value.toInt
      parse(tail)

    case ("--stopwords" | "-s") :: value :: tail =>
      numStopwords = value.toInt
      parse(tail)

    case ("--length" | "-l") :: value :: tail =>
      length = value.toInt
      parse(tail)

    case ("--boilerplate" | "-b") :: value :: tail =>
      cutoff = value.toFloat
      parse(tail)

    case ("--threshold" | "-t") :: value :: tail =>
      threshold = value.toFloat
      parse(tail)

    case ("--convergence" | "-c") :: value :: tail =>
      convergence = value.toFloat
      parse(tail)

    case ("--help" | "-h") :: tail =>
      printUsageAndExit(0)

    case _ =>
  }

  /**
   * Print usage and exit JVM with the given exit code.
   */
  private def printUsageAndExit(exitCode: Int) {
    val usage =
     s"""
      |Usage: spark-submit --class io.github.karlhigley.Summarizer <jar-path> [summarizer options]
      |
      |Options:
      |   -i PATH, --input PATH          Relative path of input files (default: "./input")
      |   -o PATH, --output PATH         Relative path of output files (default: "./output")
      |   -p VALUE, --partitions VALUE   Number of partitions for documents (default: automatic by Spark)
      |   -k VALUE, --buckets VALUE      Number of LSH buckets for documents (default: 64)
      |   -s VALUE, --stopwords VALUE    Number of stopwords to remove (default: 250)
      |   -l VALUE, --length VALUE       Number of sentences to extract from each document (default: 5) 
      |   -b VALUE, --boilerplate VALUE  Similarity cutoff for cross-document boilerplate filtering (default: 0.8)
      |   -t VALUE, --threshold VALUE    Similarity threshold for LexRank graph construction (default: 0.1)
      |   -c VALUE, --convergence VALUE  Convergence tolerance for PageRank graph computation (default: 0.001)
     """.stripMargin
    System.err.println(usage)
    System.exit(exitCode)
  }

}