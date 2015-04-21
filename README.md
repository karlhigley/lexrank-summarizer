# LexRank Summarizer

This is a Spark-based extractive summarizer, based on the [LexRank algorithm](http://arxiv.org/pdf/1109.2128.pdf).  It extracts the 5 "most informative" sentences from each document in the corpus.

## Usage

Build a JAR file from the source with `sbt assembly`.  Submit a job to Spark with:

```
spark-submit --class io.github.karlhigley.Summarizer <path to jar file> [options]

Options:
-i PATH, --input PATH          Relative path of input files (default: "./input")
-o PATH, --output PATH         Relative path of output files (default: "./output")
-s PATH, --stopwords PATH      Relative path of stopwords file (default: "./stopwords")
```

### File Formats

The summarizer expects tab-separated text files with each document on a single line.  Each line should contain a document identifier in the first column and the document text in the second column.

Outputs are formatted similarly, but with the text of a single sentence in the second column.

Stopwords are provided as a text file with one word per line.