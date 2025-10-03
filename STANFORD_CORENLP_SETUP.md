# Stanford CoreNLP Server Setup for Project Chimera

This guide explains how to set up a dedicated Stanford CoreNLP server to work with Project Chimera.

These steps and notes were written on 2025-07-01.

---

## 1. Virtual Machine (VM) Settings

- 2 CPUs  
- 4GB RAM
- Bridged Networking  
- 25GB VDI  

This is a **bare minimum** setup. Stanford CoreNLP is RAM intensive, increasing the amount of RAM will benefit you if you are able to.

8GB is required for certain annotator configurations, with 16GB being a decent allocation if your environment is able to support that.

Others may find this Medium article insightful as well: [MacOSX Setup Guide For Using Stanford CoreNLP](https://shandou.medium.com/macosx-setup-guide-for-using-stanford-corenlp-b87795b7ff4b)

---

## 2. Install Ubuntu Server 24.x

Once Ubuntu 24.x is installed, perform the below steps to get the basics installed and configured. Change the timezone to match your location:

```bash
sudo apt update
sudo apt upgrade
sudo apt install net-tools unzip
sudo timedatectl list-timezones | grep Chicago
sudo timedatectl set-timezone America/Chicago
sudo apt install unzip
```

---

## 3. Install the latest version of OpenJDK. As of 2025-07-01, that was v21:

```bash
sudo apt install openjdk-21-jdk
```

---

## 4. Download latest version of Stanford CoreNLP:

You can find the latest version here: [https://stanfordnlp.github.io/CoreNLP/download.html](https://stanfordnlp.github.io/CoreNLP/download.html)

```bash
cd /opt
sudo wget https://nlp.stanford.edu/software/stanford-corenlp-4.5.10.zip
sudo unzip stanford-corenlp-4.5.10.zip
```

Download the "English (extra)" model and the KDB model:

[https://search.maven.org/remotecontent?filepath=edu/stanford/nlp/stanford-corenlp/4.5.10/stanford-corenlp-4.5.10-models-english.jar](https://search.maven.org/remotecontent?filepath=edu/stanford/nlp/stanford-corenlp/4.5.10/stanford-corenlp-4.5.10-models-english.jar
)

```bash
cd /opt/stanford-corenlp-4.5.10
sudo wget https://search.maven.org/remotecontent?filepath=edu/stanford/nlp/stanford-corenlp/4.5.10/stanford-corenlp-4.5.10-models-english.jar -O stanford-corenlp-4.5.10-models-english.jar
sudo wget https://search.maven.org/remotecontent?filepath=edu/stanford/nlp/stanford-corenlp/4.5.10/stanford-corenlp-4.5.10-models-english-kbp.jar -O stanford-corenlp-4.5.10-models-english-kbp.jar
```

---

## 5. Edit .bashrc and put the below into it:

```bash
export CLASSPATH=""
for file in `find /opt/stanford-corenlp-4.5.10/ -name "*.jar"`; do export CLASSPATH="$CLASSPATH:`realpath $file`"; done
```

Locating the origin of the above is challenging for me; all I know is that the below **is** a mention of the above fix:

[https://github.com/smilli/py-corenlp/issues/11#issuecomment-476854581](https://github.com/smilli/py-corenlp/issues/11#issuecomment-476854581)


---

## 6. Create a bash shell script with the below inside of it. NOTE: There are **TWO filenames below** that need to have the versions match

```bash
clear
cd /opt/stanford-corenlp-4.5.10
java -mx4g -cp "stanford-corenlp-4.5.10.jar:stanford-corenlp-4.5.10-models-english.jar:*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 1200000
```

---

## 7. Create the file that is specified in the analysis.toml file, the `ner_additional_tokensregex_rules_file` file:

This file needs to reside in the root of the Stanford CoreNLP install on the server

```bash
cd /opt/stanford-corenlp-4.5.10
sudo vi ner_additional_tokensregex_rules.txt (or whatever filename is defined in the analysis.toml file)
```

Copy and paste the file from your local machine
Save the file

There is an example of what this file looks like, its formatting, etc located in the `assets/NER Rules` folder.

It should be noted that this file can be fragile in the sense that it is sensitive to EOL characters, etc.

---

## 8. Make the script executable:

```bash
sudo chmod +x <script name>
```

---

## 9. Log out of the server and log back in

---

## 10. Excute the script as a normal user:

```bash
sudo ./<name of script>
```

---

## 11. Ensure that the network information for your Standford CoreNLP server match what is in the `analysis.toml`