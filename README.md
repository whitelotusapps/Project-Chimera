# Project Chimera

**Project Chimera** is a personal audio analysis and transcription pipeline designed to process long-form audio journals. 

> ⚠️ **Note:** This project is currently tailored for my personal workflow and environment. It is actively evolving toward a more general-purpose, user-friendly tool. For now, some manual setup is required. Review the Installation and Known Limitations and Caveats sections for details.

There are two scripts to this project:

`transcribe_audio_by_date.py` and `create_audio_journal_metadata.py`.

This software is currently designed to run on a Windows PC as it makes use of [Faster Whiser](https://github.com/Purfview/whisper-standalone-win) to perform the transcription of the audio files.

The `example` folder contains an example of the kind of data that can be generated from running the `create_audio_journal_metadata.py` script.

Edit the TOML files within the `config` folder to match your environment.

Different AI models can be "dropped in" the analysis pipeline by editing the `analysis.toml` file. A variety of models are already illustrated within the included `analysis.toml`.

---

## 🚧 Project Status

This repository reflects an **in-progress personal engineering project**, not a polished, general-purpose application. Some paths and metadata are currently hard-coded, and several processes assume a specific local environment. Contributions, forks, or adaptations are welcome, but **setup steps must be followed carefully** to avoid errors.

---

## Basic Usage

You will first need to transcribe audio files using the `transcribe_audio_by_date.py` script. There is a specific format that the audio files must be named. The audio files must begin with the date and time of their recording, in the below format:

```
<YYYY-MM-DD> - <HH-MM-SS>
```

An example of a filename is:

```
2025-10-03 - 10-47-41 - audio journal.flac
```

The `transcribe_audio_by_date.py` script will rename the file. Using the above as an example, the renamed file will be:

```
2025-10-03 - 10-47-41 - 2025-10-03 - 10-49-09 - 88 - audio journal.flac
```

The `transcribe_audio_by_date.py` script uses [Faster Whiser](https://github.com/Purfview/whisper-standalone-win) and is designed to run on a Windows PC.

The `transcribe_audio_by_date.py` script will only transcribe audio files that are named for dates that are **after** the current day. Meaning that the script will **only** process audio files for days that have already passed. For example, if you have a file named `2025-10-03 - 10-47-41 - 2025-10-03 - 10-49-09 - 88 - audio journal.flac` and say the current date is October 3rd, 2025, then the script will **only** process that file **after** October 3rd, 2025 has passed. The reason for this is because part of the tagging process is to denote multiple audio files for a specific day. This is so that a "set" of audio journals can be noted and tracked per day.

Once you have audio that has been transcribed, then you can run the `create_audio_journal_metadata.py`. This script will use the JSON output created by Faster Whisper and create a new, enriched JSON file that contains analysis performed by the AI models as specified in the `analysis.toml` file.

As mentioned above:

The `example` folder contains an example of the kind of data that can be generated from running the `create_audio_journal_metadata.py` script.

Edit the TOML files within the `config` folder to match your environment.

Different AI models can be "dropped in" the analysis pipeline by editing the `analysis.toml` file. A variety of models are already illustrated within the included `analysis.toml`.

You will notice as well that there is data contained within the JSON output that is yet to be fully utilized. Part of this project was to get extremely granular in my analysis and perform microprosodic and paralinguistic acoustic analysis down to the milisecond time resolution. However, it became clear to me that my recording setup does not afford me the audio clarity required to make that a percise science.

---

## Installation

This project requires **Python 3.12+** and a few system and Python packages to run successfully. Follow the steps below to set up your environment.

### 1. Clone the repository

```bash
git clone https://github.com/whitelotusapps/project-chimera.git
cd project-chimera

pip install --upgrade pip
pip install -r requirements.txt
```

If you intend to use [Stanford CoreNLP](https://stanfordnlp.github.io/CoreNLP/), then please refer to these notes for how I got it working for this project: [STANFORD_CORENLP_SETUP.md](./STANFORD_CORENLP_SETUP.md)

---

## Known Limitations and Caveats

### 1) 📁 Required Folder Structure

Before running the scripts, you **must manually create** the following folder structure on your system. If these directories are missing, the application will raise `FileNotFoundError` exceptions or log errors.

The below is an example folder structure, the folders are specified in the `transcription.toml` and `analysis.toml` files within the `config` folder.

```
C:\temp\Transcriptions\

C:\temp\Transcriptions\raw\

C:\temp\Transcriptions\search_and_replaced\
C:\temp\Transcriptions\search_and_replaced\JSON\

C:\temp\Transcriptions\analysis\

C:\temp\code\Project Chimera\assets\temp\
```

### Why This Matters

* The `Transcriptions` subfolders are required for JSON, TEXT, and analysis output.
* The `assets\temp\` folder is used to save **word cloud images**. If it’s missing, the following kind of error will occur during tag population:

```
FileNotFoundError: [Errno 2] No such file or directory: 'C:\\temp\\code\\Project Chimera\\assets\\temp\\<filename>_wordcloud.png'
```

Automatic folder creation is planned for a future release.

---

### 2) 🪐 Swiss Ephemeris Requirement

Astrological analysis requires the **Swiss Ephemeris** data files. Download and unzip the `ephe` folder into the `swiss_eph_path` directory specified in your `analysis.toml` configuration file:

* 📥 [Swiss Ephemeris Repository](https://github.com/aloistr/swisseph)
* [Direct link to ephe folder](https://github.com/aloistr/swisseph/tree/master/ephe)

Without these files, astrological calculations will fail.

---

### 3) Zodiacal Releasing Data

There are `.tsv` fies located within the `assets/CSV` folder. These are samples of what was manually generated from the Zodiacal Releasing information generated from the [Delphic Oracle](https://www.astrology-x-files.com/delphicoracle-download.html) software by Curtis Manwaring.

---

### 4) 📝 Metadata Behavior

At present, MP3 metadata tags are **hard-coded** to use:

```
Author / Performer: The Real Zack Olinger
```

This reflects the personal nature of the project. Future updates will move these values into a configuration file.

---

### 5) 🧠 Analysis Configuration

If all analysis models in `analysis.toml` are set to `no`, the script will **raise an error** instead of exiting gracefully. This is a known behavior and will be addressed in a future version.

---

### 6) 🎧 FLAC File Support

FLAC files are **supported**, and both analysis JSON and metadata are generated successfully. However, during processing, a **single non-critical error** is raised due to a timestamp parsing issue:

```
ValueError: time data '2025:07:06 23 32-50' does not match format '%Y-%m-%d %H:%M:%S'
```

Processing and analysis is **still functional** for FLAC audio files. A fix for this parsing edge case is planned.

---

### 7) Analysis Logging

It is a known issue that the logging of the analysis is incomplete. While a logfile **is** generated, complete logging of the process is absent.

---

## 🤝 Contributing

While this is primarily a personal project, contributions are welcome! If you’d like to adapt Project Chimera for broader use, feel free to fork, open issues, or submit pull requests. Please keep in mind that some modules are currently optimized for **local, single-user workflows**.

---

## 📜 License

This work is licensed under a **Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License**.

[![CC BY-NC-SA 4.0](https://i.creativecommons.org/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

This means you are free to share and adapt this work for any non-commercial purpose, provided you give appropriate credit and distribute any derivative works under the same license. Please see the `LICENSE.md` file for the full legal code.

---

## 📚 Third-Party Data

This project includes the **XANEW dataset**, distributed under the
[Creative Commons Attribution–NonCommercial–NoDerivs 3.0 Unported License](https://creativecommons.org/licenses/by-nc-nd/3.0/).

- Source: Warriner, A.B., Kuperman, V., & Brysbaert, M. (2013). Norms of valence, arousal, and dominance for 13,915 English lemmas. *Behavior Research Methods*, 45, 1191–1207.
- License: CC BY-NC-ND 3.0

The dataset is included unmodified and is **not covered by this project’s primary license** (CC BY-NC-SA 4.0). Any use of the dataset must comply with its own license terms.

Please see the [`LICENSE-NOTICE.md`](./assets/CSV/LICENSE-NOTICE.md) file for detais.

## Conceptual Foundation

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17109163.svg)](https://doi.org/10.5281/zenodo.17109163)

This project is grounded in the methodological and conceptual framework outlined in the [Prolegomenon of Cybernetic Shamanism](https://github.com/whitelotusapps/Prolegomenon-of-Cybernetic-Shamanism). The Prolegomenon is the codification of the authors mental operating system; of which the author was assisted in becoming aware of through the use of the analytical code of Project Chimera.