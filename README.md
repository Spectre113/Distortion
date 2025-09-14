<p align="center">
  <img src="https://img.shields.io/badge/status-in%20progress-yellow.svg">
  <img src="https://img.shields.io/badge/license-MIT-blue.svg">
  <img src="https://img.shields.io/github/issues/Spectre113/Distortion" alt="GitHub issues">
  <img src="https://img.shields.io/github/last-commit/Spectre113/Distortion" alt="Last commit">
</p>

<h1 align="center">Distortion</h1>

## 📌 Overview

The main goal of our project is to help music enthusiasts to separate the soundtrack into several independent channels (such as bass, guitar, vocal) and allow them to enhance the quality of separated parts. For example, vocal actor can use just instrumental part without vocal to create its own cover.

---


## 📖 Content
- [📌 Overview](#-overview)
- [🚀 Tech Stack](#-tech-stack)
- [📁 Folder Structure](#-folder-structure)
- [🔥 Getting Started](#-getting-started)
- [📍 Waypoints](#-waypoints)
- [👥 Team](#-team)
- [📈 Roadmap](#-roadmap)
- [📝 License](#-license)

---

## 🚀 Tech Stack

| Category       | Tools / Libraries                              | Why we chose them                             |
|----------------|------------------------------------------------|-----------------------------------------------|
| Deep Learning  | Keras, TensorFlow, PyTorch     | For advanced modeling |
| Backend        | FastAPI                                      | Fast, modern, built-in OpenAPI docs           |
| Frontend       | HTML, CSS, React                    | Modern library |
| Data Processing| pandas, NumPy                              | Standard tools for loading and transforming data |
| Infrastructure | Docker, docker-compose, apache airflow, MLflow                     | Reproducibility, unified local setup, simplicity in deploying          |

## 📁 Folder Structure

```
Distortion/
├── frontend/           # Client-side application
├── backend/            # Server-side logic and API handling
├── data/               # models, scripts
├── ml/                 # Machine learning models, training, and inference scripts
├── docker-compose.yml  # Orchestration file for running all services together
├── .gitignore          # Git exclusion rules
└── README.md           # Project overview and instructions
```

---

## 🔥 Getting Started

Follow the steps below to run the project locally using Docker or manually.

### ⚙️ Prerequisites

- [Docker](https://www.docker.com/products/docker-desktop) installed and running
- Python 3.10+
- (Optional) Live server if running without Docker

### 📥 Step 1 — Clone the Repository

In your terminal:

```bash
git clone https://github.com/Spectre113/Distortion.git
cd Distortion
```

### 💻 Step 2 — Run with Docker (Recommended)

```bash
docker-compose up --build
```
This will:
- Start the **FastAPI backend** at [http://localhost:8000](http://localhost:8000)
- Start the **Frontend** at [http://localhost:3000](http://localhost:3000)

#### 📋 Managing the Container

To view logs from the running container:
```bash
docker compose logs -f
```

To stop the container without removing it:
```bash
docker compose stop
```

To stop and remove the container:
```bash
docker compose down
```

To rebuild the image after making changes to the code:
```bash
docker compose up -d --build
```

### 🛠 Alternative — Run Manually (Without Docker)
- Clone the repository (follow step 1).
- Make sure you are in the project root folder.
- Follow the steps bellow.

#### ▶️ Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
#### 🌐 Frontend
Open frontend/index.html directly in your browser or use a local server (ex. "Live Server" in VS Code)

---

## 📍 Waypoints

| Method | Endpoint                    | Function                                                              |
| -----  | --------------------------- | --------------------------------------------------------------------- |
| GET    | /upload                     | Send audio file to server                                        |
| GET    | /getFile | Get improved audio file                                                |
| GET    | /health                     | Checking if the backend is working                                    |
| GET    | /version                    | API/model version                                                     |          

## 👥 Team


|       **Name**       |                     **Responsibilities**               |      **Email**      |
|:--------------------:|:-----------------------------------------------:|:--------------------------:|
| Ilya Grigorev        | Deploy, MLOps |         il.grigorev@innopolis.university                 |
| Salavat Faizullin    | Model train, Backend      |               s.faizullin@innopolis.university           |
| Vladimir Toporkov    | Data preprocessing, Frontend, DevOps      |               v.toporkov@innopolis.university           |

---

## 📈 Roadmap
- [ ] Project structure setup
- [ ] Dataset collection
- [ ] Frontend and backend boilerplates
- [ ] Model train and deploy
- [ ] UI integration

## 📝 License

This project is licensed under the [MIT License](LICENSE).
