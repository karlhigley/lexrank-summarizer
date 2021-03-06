# LexRank Summarizer

This is a Spark-based extractive summarizer, based on the [LexRank algorithm](http://arxiv.org/pdf/1109.2128.pdf).  It extracts a 5 sentence summary from each document in the corpus.

Boilerplate sentences are detected across all documents in the corpus using frequent [sign-random projection locality-sensitive hashing](http://www.cs.princeton.edu/~moses/papers/similar.ps) (SRP-LSH) signatures, and detected within documents in the corpus using estimations of cosine similarity based on the SRP-LSH signatures.

For an explanation of the [pooling trick](https://github.com/karlhigley/lexrank-summarizer/blob/master/src/main/scala/io/github/karlhigley/lexrank/SignRandomProjectionLSH.scala#L12) used in this SRP-LSH implementation, see [Online Generation of Locality Sensitive Hash Signatures](http://www.cs.jhu.edu/~vandurme/papers/VanDurmeLallACL10.pdf).

## Usage

Build a JAR file from the source with `sbt assembly`.  Submit a job to Spark with:

```
spark-submit --class io.github.karlhigley.lexrank.Driver <path to jar file> [options]

Options:
-i PATH,  --input PATH         Relative path of input files (default: "./input")
-o PATH,  --output PATH        Relative path of output files (default: "./output")
-s VALUE, --stopwords VALUE    Number of stopwords to remove (default: 250)
-l VALUE, --length VALUE       Number of sentences to extract from each document (default: 5) 
-b VALUE, --boilerplate VALUE  Similarity cutoff for cross-document boilerplate filtering (default: 0.8)
-t VALUE, --threshold VALUE    Similarity threshold for LexRank graph construction (default: 0.1)
-c VALUE, --convergence VALUE  Convergence tolerance for PageRank graph computation (default: 0.001)
```

### File Formats

The summarizer expects tab-separated text files with each document on a single line.  Each line should contain a document identifier in the first column and the document text in the second column.

Outputs are formatted similarly, but with the text of a single sentence in the second column.
