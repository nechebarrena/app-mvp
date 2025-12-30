# App MVP: AI Video Analysis

A modular computer vision project designed to prototype, train, and validate object detection and tracking models (YOLO based) for eventual deployment on mobile devices (iOS/Android).

## Project Overview

The goal of this project is to build an MVP that:
1.  Captures short video clips (10-60s).
2.  Processes them using AI (Object Detection + Tracking).
3.  Calculates derived metrics (speed, trajectory, counts).
4.  Displays results to the user.

This repository focuses on the **AI Core** (Python) which serves as the R&D lab and processing engine before models are optimized and exported to the mobile application.

## Quick Start

### Prerequisites
*   Python 3.10+
*   [uv](https://github.com/astral-sh/uv) (for dependency management)

### Installation
1.  Clone the repository.
2.  Initialize the environment:
    ```bash
    cd ai-core
    uv sync
    ```

### Running Tools
*   **Metadata Scanner:** Automatically generate JSON sidecars for your raw videos.
    ```bash
    python ai-core/src/input_layer/metadata_scanner.py
    ```

## Documentation

*   **[Architecture Guide](docs/architecture.md):** Detailed explanation of the project structure, Domain-Driven Design approach, and data flow.
*   **[Tooling Guide](docs/tools.md):** How to use the helper scripts (like the Metadata Scanner).

## Directory Structure
See [docs/architecture.md](docs/architecture.md) for the full breakdown.

*   `ai-core/`: Python source code.
*   `data/`: Local storage for videos and models (not synced to Git).
*   `mobile-app/`: Placeholder for the mobile client.

