Author: Mariia Fedorova
Course: AIML427 — Big Data
Assignment 3 — Individual Component

## 1. Prerequisites

These instructions assume you're working on the ECS Hadoop/Spark cluster (e.g., `co246a-1.ecs.vuw.ac.nz`) and that your environment has Spark and HDFS available.

You must also have:
- Uploaded `heart_2022_no_nans.csv` to your HDFS folder. Uploaded python scripts to the same folder. We used scp to copy from our local machines, where these files are stored:

 ```console
foo@bar:~$ scp heart_2022_no_nans.csv insert_correct_username@barretts.ecs.vuw.ac.nz:~
foo@bar:~$ scp model.py insert_correct_username@barretts.ecs.vuw.ac.nz:~
```

Transfer data file to the hadoop cluster. Shh to the cluster with HDFS and Spark:

 ```console
foo@bar:~$ ssh co246a-1
```

Check that env is configured:

```console
foo@bar:~$ source HadoopSetup.csh
foo@bar:~$ need java8
foo@bar:~$ hdfs dfs -mkdir -p /user/insert_correct_username/individual
foo@bar:~$ hdfs dfs -put heart_2022_no_nans.csv /user/insert_correct_username/individual/
```

## 2. Verify the dataset is successfully transferred:

```console
foo@bar:~$ hdfs dfs -ls /user/insert_correct_username/individual
```

You shall see the `heart_2022_no_nans.csv` if everything is correct.

Fix the path INSIDE the script `model.py`. Find the line:

```console
spark.read.csv("hdfs:///user/insert_correct_username/individual/heart_2022_no_nans.csv", header=True, inferSchema=True)
```

and insert valid path.

## 3. Activate Spark env

Since it is not described in tutorial, I needed to activate Spark myself:

```console
foo@bar:~$ export SPARK_HOME=/local/spark
foo@bar:~$ export PATH=$SPARK_HOME/bin:$PATH
```

To verify that Spark is installed and working in the env:

```console
foo@bar:~$ spark-submit --version
```

If everything is up and running, you will see something like:

```console
Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /___/ .__/\_,_/_/ /_/\_\   version 3.5.1
      /_/

Using Scala version 2.12.18, Java HotSpot(TM) 64-Bit Server VM, 1.8.0_172
```

## 4. Submit Spark job

```console
foo@bar:~$ spark-submit model.py
```