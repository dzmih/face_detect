````markdown
# sentinel ai - face recognition system v2

this is a high-performance biometric identification system built with **python 3.13**, **deepface (arcface)**, and **sqlite**. it features real-time face tracking, user registration, and detailed access logging.

## features

- **biometric id:** powered by arcface for top-tier accuracy.
- **real-time tracking:** fast face detection with opencv cascades.
- **data persistence:** all logs are stored in a structured sqlite database.
- **garbage filter:** automated frame quality check (blur and darkness detection).
- **easy deployment:** fully dockerized environment.

---

## quick start (docker)

make sure u have **docker** and **docker-compose** installed on ur machine.

1. **clone the repo:**
   ```bash
   git clone [https://github.com/ur-username/sentinel-ai.git](https://github.com/ur-username/sentinel-ai.git)
   cd sentinel-ai
   ```
````

2. **prepare the database:**
   create a folder named `db` (this is where user photos will be stored).

```bash
mkdir db

```

3. **build and run:**

```bash
docker-compose up --build

```

> **note for windows users:** running gui apps and webcam inside docker on windows is tricky and requires an x-server (like vcxsrv). if u just want to test the logic, running it natively via python is recommended.

---

## native installation (without docker)

if u want to run it directly on ur os:

1. **install dependencies:**

```bash
pip install -r requirements.txt

```

2. **run the app:**

```bash
python main.py

```

---

## how it works

1. **init:** the app loads the arcface model and connects to `security_log.db`.
2. **detection:** it looks for faces in the video stream (60 fps).
3. **verification:** every few seconds, it compares the face with embeddings in the `db/` folder.
4. **logging:** if a match is found, it logs the timestamp, name, and confidence level to sqlite.

## license

mit. feel free to use it for ur own stuff.

```

```
