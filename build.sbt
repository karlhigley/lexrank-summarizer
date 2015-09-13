name := "lexrank-summarizer"

version := "0.0.1"

scalaVersion := "2.10.4"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % "1.5.0" % "provided",
  "org.apache.spark" %% "spark-mllib" % "1.5.0" % "provided",
  "org.scalanlp" %% "breeze" % "0.11.2",
  "org.scalanlp" %% "chalk" % "1.3.2" exclude ("com.typesafe.sbt", "sbt-pgp"),
  "org.scalatest" %% "scalatest" % "2.2.4" % "test",
  "com.typesafe.akka" %% "akka-actor" % "2.3.4" % "test"
)

resolvers ++= Seq(
  "Akka Repository" at "http://repo.akka.io/releases/",
  "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/",
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

parallelExecution in Test := false
